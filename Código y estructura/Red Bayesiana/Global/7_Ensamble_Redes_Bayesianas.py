"""Ensamble probabilístico de redes bayesianas con validación anidada por sujeto.

Tres representaciones: BCG global, BCG regional y TAN/discreta. La selección
Top/intervalos/modelo, calibración y pesos solo usan sujetos de entrenamiento.
LOSO externo sobre los 32 sujetos de desarrollo y GroupKFold interno (4 por
defecto). Ajuste final en 32; los ocho externos se predicen una sola vez al final.

Reutiliza funciones de los scripts vecinos mediante importlib, sin ejecutar sus
programas principales ni modificar sus resultados. No recibe Tarea como predictor
ni la usa para enrutar épocas. Promedios de probabilidades, no unión de aristas,
no efectos causales ni promedio bayesiano formal de evidencias marginales.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from tqdm.auto import tqdm

VERSION = 'ensamble_probabilistico_anidado_v1'
ROLES = ['Global', 'Regional', 'Discreto']
ARCHIVOS = {
    'global': '1_Red_Bayesiana_Condicional_Gaussiana.py',
    'regional': '4_Red_BCG_regional.py',
    'tan': '3_Red_TAN.py',
    'discreta': '2_Red_Discreta.py',
}


def guardar_json(ruta, datos):
    ruta = Path(ruta); ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix+'.tmp')
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(temporal, ruta)


def guardar_csv(ruta, datos):
    ruta = Path(ruta); ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix+'.tmp')
    datos.to_csv(temporal, index=False)
    os.replace(temporal, ruta)


def sha_archivo(ruta):
    firma = hashlib.sha256()
    with Path(ruta).open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024*1024), b''):
            firma.update(bloque)
    return firma.hexdigest()


def cargar_modulos(carpeta, usar_discreta):
    modulos, firmas = {}, {}
    for nombre, archivo in ARCHIVOS.items():
        if nombre == 'discreta' and not usar_discreta:
            continue
        ruta = carpeta/archivo
        if not ruta.is_file():
            raise FileNotFoundError(f'Falta un componente del ensamble: {ruta}')
        especificacion = importlib.util.spec_from_file_location(f'ensamble_bn_{nombre}', ruta)
        modulo = importlib.util.module_from_spec(especificacion)
        sys.modules[especificacion.name] = modulo
        especificacion.loader.exec_module(modulo)
        modulos[nombre] = modulo
        firmas[archivo] = sha_archivo(ruta)
    modulos['regional'].cargar_pc_causal_learn()
    return modulos, firmas


def pesos_sujetos(grupos):
    grupos = np.asarray(grupos)
    if len(grupos) == 0 or pd.isna(grupos).any():
        raise ValueError('Sujetos vacíos o sin identificar.')
    cuentas = pd.Series(grupos).value_counts()
    w = np.array([1./cuentas[g] for g in grupos], dtype=float)
    return w/w.sum()


def validar_probabilidades(prob, n):
    p = np.asarray(prob, dtype=float)
    if p.ndim != 1 or len(p) != n or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('El componente no produjo una probabilidad válida por época.')
    return p


def balanceada_sujetos(y, p, grupos):
    pred = (np.asarray(p)>=.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1], sample_weight=pesos_sujetos(grupos)).ravel()
    if tp+fn == 0 or tn+fp == 0:
        raise ValueError('La evaluación requiere ambas clases.')
    return float(.5*(tp/(tp+fn)+tn/(tn+fp)))


def candidatos(args):
    banco = [{'id':f'Global_PC_Top{top}', 'rol':'Global', 'tipo':'global', 'top':top,
              'algoritmo':'PC'} for top in args.tops]
    banco.append({'id':'Regional_PC_25', 'rol':'Regional', 'tipo':'regional', 'top':25,
                  'algoritmo':'PC'})
    for tipo in args.candidatos_discretos:
        for top in args.tops:
            for bins in args.bins:
                banco.append({'id':f'{tipo}_Top{top}_Bins{bins}', 'rol':'Discreto',
                              'tipo':'tan' if tipo=='TAN' else 'discreta', 'top':top,
                              'bins':bins, 'algoritmo':'TAN' if tipo=='TAN' else 'PC'})
    return banco


def preparar_global(train, test, candidatas, seleccionador, top):
    X = train[candidatas].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    Z = test[candidatas].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    medianas = X.median()
    if medianas.isna().any():
        raise ValueError(f'Variables sin datos finitos: {medianas[medianas.isna()].index.tolist()}')
    base = pd.concat([X.fillna(medianas), train[['Ansiedad']].astype(int)], axis=1).copy()
    elegidas = seleccionador.seleccionar_top_dt(base, candidatas, 'Ansiedad', top)
    a = base[elegidas+['Ansiedad']].copy()
    b = Z[elegidas].fillna(medianas[elegidas]).copy()
    if not b.index.equals(test.index):
        raise ValueError('El preprocesamiento global alteró el orden de evaluación.')
    return a, b, {'Caracteristicas_DT':elegidas, 'Medianas':medianas[elegidas].to_dict(),
                  'Imputados_train':int(X[elegidas].isna().sum().sum()),
                  'Imputados_evaluacion':int(Z[elegidas].isna().sum().sum())}


def ajustar_base(train, test, candidato, modulos, candidatas, args, contexto):
    # Los adaptadores reciben únicamente predictores, identificador y etiqueta
    # de entrenamiento. Las etiquetas de evaluación no se pasan a los modelos.
    test = test.drop(columns=['Ansiedad','Puntaje','Tarea','Trial','Epoca'], errors='ignore')
    tipo = candidato['tipo']
    if tipo == 'global':
        m = modulos['global']
        a, b, prepro = preparar_global(train, test, candidatas, modulos['tan'], candidato['top'])
        modelo = m.ClasificadorBayesianoCondicionalGaussiano('Ansiedad', args.ridge, args.piso_varianza)
        centradas = modelo.preparar_entrenamiento(a)
        with m.mostrar_actividad(contexto):
            dag, detalles = m.aprender_estructura(centradas, 'PC', 'causal-learn', None)
        modelo.ajustar_parametros(dag)
        p = modelo.predecir_probabilidades(b)
        modulos['tan'].validar_dag(modelo.dag, a.columns)
        if modelo.dag.in_degree('Ansiedad') or set(modelo.dag.successors('Ansiedad')) != set(b.columns):
            raise ValueError('La BCG global no conservó su clase raíz.')
        evidencia = {'Preprocesamiento':prepro, 'Detalles':detalles,
                     'Modelo':{'Nodos':list(modelo.dag.nodes()),'Arcos':list(modelo.dag.edges()),
                               **modelo.dag.graph['modelo_probabilistico']}}
    elif tipo == 'regional':
        opciones = SimpleNamespace(ridge=args.ridge, piso_varianza=args.piso_varianza,
                                   pc_alpha=.05, pc_max_cond_vars=4, ejecucion_id=args.ejecucion_id)
        p, _, _, evidencia = modulos['regional'].ajustar_y_predecir(train, test, 'PC', opciones, contexto)
    else:
        opciones = SimpleNamespace(max_padres=3, max_celdas_cpd=256,
                                   diagnosticos_estructura=False, ejecucion_id=args.ejecucion_id)
        m = modulos['tan' if tipo=='tan' else 'discreta']
        p, _, _, evidencia = m.ajustar_y_predecir(train, test, candidatas, candidato['top'],
                                                candidato['bins'], candidato['algoritmo'], opciones, contexto)
    return validar_probabilidades(p, len(test)), evidencia


def predecir_base(train, test, candidato, motor, contexto):
    grupos_train = set(map(str,train.Sujeto)); grupos_test = set(map(str,test.Sujeto))
    if grupos_train & grupos_test:
        raise ValueError(f'Sujetos compartidos entre entrenamiento y evaluación: {contexto}')
    if set(train.Ansiedad.unique()) != {0,1}:
        raise ValueError(f'Falta una clase en entrenamiento: {contexto}')
    firma = {'protocolo':motor['firma'], 'candidato':candidato,
             'indices_train':list(map(str,train.index)), 'indices_test':list(map(str,test.index))}
    clave = hashlib.sha256(json.dumps(firma,sort_keys=True).encode()).hexdigest()
    ruta = motor['cache']/f'{clave}.json'
    if ruta.exists() and not motor['args'].sin_cache:
        entrada = json.loads(ruta.read_text())
        if entrada.get('Firma') != firma:
            raise ValueError(f'Firma de caché incompatible: {ruta}')
        if entrada.get('Sujetos_train') != sorted(grupos_train) or entrada.get('Sujetos_test') != sorted(grupos_test):
            raise ValueError(f'Sujetos de caché incompatibles: {ruta}')
        p = validar_probabilidades(entrada['Probabilidades'], len(test))
        return p, {'Cache':True,'Archivo':str(ruta),'Tiempo_s':0.,'Candidato':candidato['id']}
    inicio = time.perf_counter()
    p, evidencia = ajustar_base(train,test,candidato,motor['modulos'],motor['candidatas'],motor['args'],contexto)
    entrada = {'Firma':firma, 'ID_entrenamiento_origen':motor['args'].ejecucion_id,
               'Sujetos_train':sorted(grupos_train), 'Sujetos_test':sorted(grupos_test),
               'Probabilidades':p.tolist(),'Modelo_y_preprocesamiento':evidencia,
               'Tiempo_ajuste_y_prediccion_s':time.perf_counter()-inicio}
    guardar_json(ruta, entrada)
    return p, {'Cache':False,'Archivo':str(ruta),'Tiempo_s':entrada['Tiempo_ajuste_y_prediccion_s'],
               'Candidato':candidato['id']}


def calibrar(P, y, grupos, modo):
    modelos = []
    for j in range(P.shape[1]):
        if modo == 'ninguna':
            modelos.append({'Tipo':'Identidad'})
            continue
        x = np.log(np.clip(P[:,j],1e-6,1-1e-6)/np.clip(1-P[:,j],1e-6,1-1e-6)).reshape(-1,1)
        clf = LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            clf.fit(x,y,sample_weight=pesos_sujetos(grupos)*len(y))
        a, b = float(clf.coef_[0,0]), float(clf.intercept_[0])
        if not np.isfinite([a,b]).all():
            raise ValueError('Calibración no finita.')
        modelos.append({'Tipo':'Sigmoide_sobre_logit','Pendiente':a,'Intercepto':b,
                        'C':1.,'Clip_logit':1e-6,'Fuente':'OOF interna, solo sujetos de entrenamiento; calibración supervisada de probabilidades, no de EEG'})
    return modelos


def aplicar_calibracion(P, modelos):
    salida = np.empty_like(P,dtype=float)
    for j,m in enumerate(modelos):
        if m['Tipo']=='Identidad':
            salida[:,j]=P[:,j]
        else:
            x=np.log(np.clip(P[:,j],1e-6,1-1e-6)/np.clip(1-P[:,j],1e-6,1-1e-6))
            salida[:,j]=expit(m['Pendiente']*x+m['Intercepto'])
        validar_probabilidades(salida[:,j],len(P))
    return salida


def ajustar_pesos(P,y,grupos,regularizacion):
    w0=np.full(P.shape[1],1./P.shape[1]);sw=pesos_sujetos(grupos)
    def objetivo(w):
        residuo=P@w-y
        return float(np.sum(sw*residuo**2)+regularizacion*np.sum((w-w0)**2))
    def gradiente(w):
        return 2*P.T@(sw*(P@w-y))+2*regularizacion*(w-w0)
    opt=minimize(objetivo,w0,jac=gradiente,method='SLSQP',bounds=[(0.,1.)]*len(w0),
                 constraints=[{'type':'eq','fun':lambda w:float(w.sum()-1.),'jac':lambda w:np.ones_like(w)}],
                 options={'ftol':1e-11,'maxiter':1000})
    if not opt.success or not np.isfinite(opt.x).all() or abs(opt.x.sum()-1)>1e-6 or np.min(opt.x)<-1e-8:
        raise RuntimeError(f'Falló la optimización de pesos: {opt.message}')
    w=np.clip(opt.x,0.,1.);w/=w.sum()
    return w, {'Objetivo':'Brier con igual peso total por sujeto y regularización hacia pesos iguales',
               'Regularizacion':regularizacion,'Valor_objetivo':objetivo(w),'Iteraciones':int(opt.nit),
               'Pesos':dict(zip(ROLES,w.tolist()))}


def complementariedad(P,y,grupos,nombres):
    w=pesos_sujetos(grupos);err=(P>=.5).astype(int)!=np.asarray(y)[:,None];pares=[]
    for a in range(len(nombres)):
        for b in range(a+1,len(nombres)):
            corr=float(np.corrcoef(P[:,a],P[:,b])[0,1]) if np.std(P[:,a])>0 and np.std(P[:,b])>0 else None
            pares.append({'Modelo_A':nombres[a],'Modelo_B':nombres[b],'Correlacion_probabilidades':corr,
                          'Ambos_se_equivocan':float(w@(err[:,a]&err[:,b])),
                          'A_acierta_B_falla':float(w@(~err[:,a]&err[:,b])),
                          'B_acierta_A_falla':float(w@(~err[:,b]&err[:,a])),
                          'Desacuerdo_clasificacion':float(w@(err[:,a]!=err[:,b]))})
    return {'Pares':pares,'Ponderacion':'Igual peso total por sujeto; descriptivo, sin decisión sobre prueba externa'}


def variantes(P,Q,meta,args):
    salida={f'{rol}_original':P[:,j] for j,rol in enumerate(ROLES)}
    salida['Mejor_individual_interno']=P[:,meta['Indice_mejor_original']]
    salida['Promedio_simple']=P.mean(axis=1)
    if args.calibracion!='ninguna':
        salida.update({f'{rol}_calibrado':Q[:,j] for j,rol in enumerate(ROLES)})
        salida['Mejor_individual_calibrado']=Q[:,meta['Indice_mejor_calibrado']]
        salida['Promedio_calibrado']=Q.mean(axis=1)
    salida['Ensamble_ponderado']=Q@np.asarray(meta['Pesos'])
    return {k:validar_probabilidades(v,len(P)) for k,v in salida.items()}


def ajustar_entrenamiento(train,banco,motor,contexto):
    args=motor['args'];cv=GroupKFold(n_splits=args.pliegues_internos)
    particiones=list(cv.split(train,train.Ansiedad,train.Sujeto));prob={};resultados=[];refs={}
    carpeta=motor['evidencia']/'Seleccion_interna'/contexto
    for candidato in tqdm(banco,desc=f'{contexto}: candidatos internos',unit='modelo'):
        p=np.full(len(train),np.nan);referencias=[];error=None
        for fold,(it,iv) in enumerate(particiones,1):
            etiqueta=f'{contexto}, interno {fold}/{len(particiones)}, {candidato["id"]}'
            try:
                pred,ref=predecir_base(train.iloc[it],train.iloc[iv],candidato,motor,etiqueta)
                p[iv]=pred;referencias.append({'Fold_interno':fold,**ref,
                  'Sujetos_train':sorted(set(map(str,train.iloc[it].Sujeto))),
                  'Sujetos_validacion':sorted(set(map(str,train.iloc[iv].Sujeto)))})
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                tqdm.write(f'CANDIDATO_NO_DISPONIBLE {etiqueta}: {error}')
                guardar_json(carpeta/f'Errores/{candidato["id"]}.json',
                             {'Contexto':etiqueta,'Error':error,'ID_ejecucion':args.ejecucion_id})
                break
        fila={'ID_ejecucion':args.ejecucion_id,**candidato,'Estado':'NO_DISPONIBLE' if error else 'COMPLETO','Error':error,
              'Pliegues_completos':len(referencias)}
        if error is None:
            validar_probabilidades(p,len(train))
            fila['Balanced_accuracy_interna']=balanceada_sujetos(train.Ansiedad.to_numpy(),p,train.Sujeto.to_numpy())
            fila['Brier_interno']=float(pesos_sujetos(train.Sujeto)@((p-train.Ansiedad.to_numpy())**2))
            prob[candidato['id']]=p;refs[candidato['id']]=referencias
        resultados.append(fila)
        guardar_csv(carpeta/'Candidatos.csv',pd.DataFrame(resultados))
    elegidos=[]
    for rol in ROLES:
        validos=[f for f in resultados if f['rol']==rol and f['Estado']=='COMPLETO']
        if not validos:
            raise RuntimeError(f'No hay candidato completo para {rol} en {contexto}; revise los errores guardados.')
        # Empates: menos características, menos intervalos y nombre determinista.
        validos.sort(key=lambda f:(-f['Balanced_accuracy_interna'],f['top'],f.get('bins',0),f['id']))
        elegidos.append(next(c for c in banco if c['id']==validos[0]['id']))
    P=np.column_stack([prob[c['id']] for c in elegidos]);y=train.Ansiedad.to_numpy(dtype=int);grupos=train.Sujeto.to_numpy()
    calibradores=calibrar(P,y,grupos,args.calibracion);Q=aplicar_calibracion(P,calibradores)
    pesos,opt=ajustar_pesos(Q,y,grupos,args.regularizacion_pesos)
    original=[balanceada_sujetos(y,P[:,j],grupos) for j in range(3)]
    calibrado=[balanceada_sujetos(y,Q[:,j],grupos) for j in range(3)]
    meta={'ID_ejecucion':args.ejecucion_id,'Sujetos_entrenamiento':sorted(set(map(str,grupos))),
          'Seleccionados':elegidos,'Calibradores':calibradores,'Pesos':pesos.tolist(),'Optimizacion':opt,
          'Indice_mejor_original':int(np.argmax(original)),'Indice_mejor_calibrado':int(np.argmax(calibrado)),
          'Referencia_mejor_original':elegidos[int(np.argmax(original))]['id'],
          'Balanced_accuracy_interna_original':dict(zip(ROLES,original)),
          'Balanced_accuracy_interna_calibrada':dict(zip(ROLES,calibrado)),
          'Ajustes_internos_seleccionados':{c['id']:refs[c['id']] for c in elegidos},
          'N_candidatos_no_disponibles':sum(f['Estado']!='COMPLETO' for f in resultados),
          'Nota':'Las métricas internas sirven para selección y ajuste. El rendimiento se evalúa en el sujeto externo al entrenamiento; '
                 'las métricas calibradas internas no son una evaluación independiente.'}
    tabla=pd.DataFrame({'Indice_fila':train.index,'Sujeto':grupos,'Ansiedad_real':y})
    for j,rol in enumerate(ROLES):tabla[f'Prob_{rol}']=P[:,j];tabla[f'Prob_{rol}_calibrada']=Q[:,j]
    guardar_csv(carpeta/'Predicciones_OOF_internas_seleccionadas.csv',tabla)
    guardar_json(carpeta/'Seleccion_y_pesos.json',meta)
    guardar_json(carpeta/'Complementariedad.json',complementariedad(P,y,grupos,ROLES))
    return meta


def predecir_entrenamiento(train,test,meta,motor,contexto):
    probabilidades=[];modelos=[]
    for candidato in meta['Seleccionados']:
        p,ref=predecir_base(train,test,candidato,motor,f'{contexto}, reajuste {candidato["id"]}')
        probabilidades.append(p);modelos.append(ref)
    P=np.column_stack(probabilidades);Q=aplicar_calibracion(P,meta['Calibradores'])
    return variantes(P,Q,meta,motor['args']),modelos


def tabla_predicciones(datos,predicciones,run):
    tabla=pd.DataFrame({'ID_ejecucion':run,'Indice_fila':datos.index.to_numpy(),
                        'Sujeto':datos.Sujeto.to_numpy(),'Ansiedad_real':datos.Ansiedad.to_numpy(dtype=int)})
    for nombre,p in predicciones.items():
        tabla[f'Prob_{nombre}']=validar_probabilidades(p,len(datos))
        tabla[f'Pred_{nombre}']=(p>=.5).astype(int)
    return tabla


def resumir(datos,predicciones,modulo,fase,completo=True):
    filas=[];y=datos.Ansiedad.to_numpy(dtype=int);grupos=datos.Sujeto.to_numpy()
    for nombre,p in predicciones.items():
        pred=(p>=.5).astype(int);fila={'Modelo':nombre,'Estado':'COMPLETO' if completo else 'EN_CURSO',
                                    f'N_epocas ({fase})':len(datos),f'N_sujetos ({fase})':len(set(grupos)),'Umbral_fijo':.5}
        fila.update({f'{k} ({fase})':v for k,v in modulo.metricas_clasificacion(y,pred,p).items()})
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        fila.update({f'{k} ({fase})':int(v) for k,v in zip(['TN','FP','FN','TP'],[tn,fp,fn,tp])})
        fila.update(modulo.metricas_agregadas_por_sujeto(y,pred,grupos,fase))
        if completo and len(set(grupos))>=2:fila.update(modulo.bootstrap_ci_por_sujeto(y,pred,grupos,fase))
        filas.append(fila)
    return pd.DataFrame(filas)


def graficas(salida,val,test,yval,ytest,pesos,final):
    etiquetas=['Global','Regional','Discreto']
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(etiquetas,final['Pesos'],color=['#376d9c','#609e69','#c68948'])
    ax.set_ylim(0,1);ax.set_ylabel('Peso');ax.set_title('Pesos finales ajustados con desarrollo')
    fig.tight_layout();fig.savefig(salida/'Pesos_finales.png',dpi=250);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for j,rol in enumerate(ROLES):ax.plot(range(1,len(pesos)+1),np.asarray(pesos)[:,j],label=rol)
    ax.set_ylim(0,1);ax.set_xlabel('Pliegue LOSO');ax.set_ylabel('Peso');ax.legend();ax.set_title('Variación de pesos entre sujetos reservados')
    fig.tight_layout();fig.savefig(salida/'Pesos_LOSO.png',dpi=250);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    nombres=['Global_original','Regional_original','Discreto_original','Promedio_simple','Ensamble_ponderado']
    for ax,d,y,titulo in [(axes[0],val,yval,'LOSO anidado'),(axes[1],test,ytest,'Ocho sujetos externos')]:
        for nombre in nombres:
            fpr,tpr,_=roc_curve(y,d[nombre]);ax.plot(fpr,tpr,label=nombre.replace('_',' '))
        ax.plot([0,1],[0,1],'--',color='grey');ax.set_title(titulo);ax.set_xlabel('Tasa de falsos positivos');ax.set_ylabel('Sensibilidad');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(salida/'Curvas_ROC.png',dpi=250);plt.close(fig)


def diagnostico_calibracion(salida, evidencia, val, test, yval, ytest):
    nombres=['Global_original','Regional_original','Discreto_original','Promedio_simple','Ensamble_ponderado']
    filas=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    limites=np.linspace(0.,1.,6)
    for ax,pred,y,fase in [(axes[0],val,yval,'Val'),(axes[1],test,ytest,'Test')]:
        for nombre in nombres:
            p=pred[nombre];x=[];z=[]
            for b in range(5):
                mascara=(p>=limites[b]) & ((p<limites[b+1]) if b<4 else (p<=1.))
                n=int(mascara.sum())
                media=float(np.mean(p[mascara])) if n else None
                frecuencia=float(np.mean(y[mascara])) if n else None
                filas.append({'Fase':fase,'Modelo':nombre,'Limite_inferior':float(limites[b]),
                              'Limite_superior':float(limites[b+1]),'N_epocas':n,
                              'Probabilidad_media':media,'Proporcion_positiva_observada':frecuencia})
                if n:x.append(media);z.append(frecuencia)
            ax.plot(x,z,marker='o',label=nombre.replace('_',' '))
        ax.plot([0,1],[0,1],'--',color='grey');ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_xlabel('Probabilidad media');ax.set_ylabel('Proporción positiva observada')
        ax.set_title('LOSO anidado' if fase=='Val' else 'Ocho sujetos externos');ax.legend(fontsize=8)
    guardar_csv(evidencia/'Calibracion_por_intervalos.csv',pd.DataFrame(filas))
    fig.tight_layout();fig.savefig(salida/'Calibracion_probabilidades.png',dpi=250);plt.close(fig)


def principal(argv=None):
    parser=argparse.ArgumentParser(description='Ensamble de redes bayesianas, LOSO y selección/calibración/pesos internos por sujeto.')
    parser.add_argument('--tops',nargs='+',type=int,choices=[10,15,20],default=[10,15,20])
    parser.add_argument('--bins',nargs='+',type=int,choices=[2,3],default=[2,3])
    parser.add_argument('--candidatos-discretos',nargs='+',choices=['TAN','Discreta'],default=['TAN','Discreta'])
    parser.add_argument('--pliegues-internos',type=int,default=4)
    parser.add_argument('--calibracion',choices=['sigmoide','ninguna'],default='sigmoide')
    parser.add_argument('--regularizacion-pesos',type=float,default=.001)
    parser.add_argument('--ridge',type=float,default=1.)
    parser.add_argument('--piso-varianza',type=float,default=1e-6)
    parser.add_argument('--sin-cache',action='store_true',help='Recalcula ajustes aunque exista una caché compatible.')
    args=parser.parse_args(argv);args.ejecucion_id=uuid4().hex
    if not 2<=args.pliegues_internos<=10:parser.error('pliegues-internos debe estar entre 2 y 10.')
    if not np.isfinite([args.regularizacion_pesos,args.ridge,args.piso_varianza]).all() or min(args.regularizacion_pesos,args.ridge)<0 or args.piso_varianza<=0:
        parser.error('Regularización y ridge deben ser no negativos; piso-varianza positivo; todos finitos.')
    args.tops=list(dict.fromkeys(args.tops));args.bins=list(dict.fromkeys(args.bins));args.candidatos_discretos=list(dict.fromkeys(args.candidatos_discretos))
    carpeta=Path(__file__).resolve().parent;raiz=carpeta.parents[2]
    modulos,fuentes=cargar_modulos(carpeta,'Discreta' in args.candidatos_discretos)
    desarrollo,externa,candidatas,sujetos,excluidas=modulos['tan'].cargar_datos(raiz)
    # Tarea y otras anotaciones del experimento se eliminan antes de cualquier ajuste.
    if set(candidatas) & {'Tarea','Trial','Epoca','Puntaje','Ansiedad','Sujeto'}:
        raise ValueError('La lista de características incluye anotaciones del experimento o etiquetas.')
    necesarias=list(dict.fromkeys(candidatas+modulos['regional'].COLUMNAS_ORIGEN+['Sujeto','Ansiedad']))
    desarrollo=desarrollo[necesarias].copy();externa=externa[necesarias].copy()
    salida=raiz/'Resultados'/'Red Bayesiana'/'Global'/'Ensamble_Probabilistico';ev=salida/'Evidencia'
    salida.mkdir(parents=True,exist_ok=True)
    fuentes[Path(__file__).name]=sha_archivo(__file__)
    protocolo={'Version':VERSION,'Fuentes':fuentes,'Datos_sha256':sha_archivo(raiz/'Resultados/Exploratorio/datos_completos_normalizados.parquet'),
               'Candidatas':candidatas,'Ridge':args.ridge,'Piso_varianza':args.piso_varianza,
               'Versiones':{p:modulos['tan'].obtener_version(p) for p in ['pgmpy','causal-learn','numpy','pandas','scikit-learn','scipy']}}
    firma=hashlib.sha256(json.dumps(protocolo,sort_keys=True).encode()).hexdigest()
    motor={'args':args,'modulos':modulos,'candidatas':candidatas,'evidencia':ev,'firma':firma,'cache':ev/'Cache'/firma}
    banco=candidatos(args);inicio=time.perf_counter()
    config={**vars(args),'ID_ejecucion':args.ejecucion_id,'Version_modelo':VERSION,'Estado':'EN_CURSO',
            'Inicio_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'Protocolo_cache':protocolo,
            'Sujetos_desarrollo_32':list(map(str,sujetos[:32])),'Sujetos_externos_8':list(map(str,sujetos[32:])),
            'Epocas_excluidas_por_criterio_SAM':excluidas,'Criterio':'SAM=0 control; SAM>=1 caso','Umbral_fijo':.5,
            'Banco_candidatos':banco,'Nota_seleccion_componentes':'La selección de componentes se realiza con probabilidades originales; calibración y pesos se ajustan después, siempre dentro del entrenamiento.',
            'Seleccion':'Exactitud balanceada interna con igual peso por sujeto; desempate por menor Top/intervalos e identificador',
            'Pesos':'No negativos, suma uno; Brier interno y regularización hacia pesos iguales; sin optimización con sujetos externos',
            'Validacion':'LOSO externo sobre desarrollo; GroupKFold interno; calibradores entrenados en OOF internas de cada entrenamiento',
            'Calibracion':'Sigmoide sobre logit, C=1, pesos por sujeto; ajuste en predicciones OOF internas seleccionadas. Opción ninguna disponible.',
            'Comparadores':'Componentes originales y calibrados, mejor individual elegido internamente y promedios simples',
            'Limites':'Las métricas internas se usan para seleccionar. Ocho externos ya observados anteriormente; comparación conocida, no nueva confirmación independiente.',
            'Tipos_modelos':'BCG global PC/pgmpy; BCG regional PC/causal-learn; TAN o discreta PC/pgmpy. Sin información Tarea.',
            'Interpretacion':'Ensamble predictivo de redes, no una única red con arcos fusionados ni inferencia causal',
            'Versiones':{p:modulos['tan'].obtener_version(p) for p in ['pgmpy','causal-learn','numpy','pandas','scikit-learn','scipy']}}
    guardar_json(ev/'Configuracion.json',config)
    resumen=salida/'Resultados_Ensamble_Probabilistico.csv'
    guardar_csv(resumen,pd.DataFrame(columns=['Modelo','Estado']))
    max_ajustes=32*(args.pliegues_internos*len(banco)+3)+args.pliegues_internos*len(banco)+3
    print(f'Resultados: {salida}',flush=True)
    print(f'{len(banco)} candidatos internos, {args.pliegues_internos} particiones por sujeto; hasta {max_ajustes} ajustes base sin caché.',flush=True)
    print('No se leen resultados históricos para seleccionar; la selección se recalcula dentro de cada entrenamiento.',flush=True)
    predicciones={};pesos=[];selecciones=[];contexto='Inicio';estado={'ID_ejecucion':args.ejecucion_id,'Estado':'EN_CURSO','Pliegues':[]}
    guardar_json(ev/'Estado_LOSO.json',estado)
    columnas_pred=['ID_ejecucion','Indice_fila','Sujeto','Ansiedad_real']
    guardar_csv(ev/'Predicciones_LOSO.csv',pd.DataFrame(columns=columnas_pred))
    try:
        particiones=LeaveOneGroupOut().split(desarrollo,desarrollo.Ansiedad,desarrollo.Sujeto)
        for fold,(it,iv) in enumerate(tqdm(particiones,total=32,desc='LOSO anidado del ensamble',unit='sujeto'),1):
            sub=str(desarrollo.iloc[iv].Sujeto.iloc[0]);contexto=f'Fold_{fold:02d}_Sujeto_{sub}'
            train,test=desarrollo.iloc[it],desarrollo.iloc[iv]
            if len(set(train.Sujeto))!=31 or len(set(test.Sujeto))!=1:raise ValueError('LOSO no conserva 31/1 sujetos.')
            inicio_fold=time.perf_counter();meta=ajustar_entrenamiento(train,banco,motor,contexto)
            salidas,refs=predecir_entrenamiento(train,test,meta,motor,contexto)
            if not predicciones:predicciones={n:np.full(len(desarrollo),np.nan) for n in salidas}
            for nombre,p in salidas.items():predicciones[nombre][iv]=p
            pesos.append(meta['Pesos']);selecciones.extend(meta['Seleccionados'])
            registro={'Fold':fold,'Sujeto_validacion':sub,'Tiempo_s':time.perf_counter()-inicio_fold,'Seleccion_y_pesos':meta,'Modelos_evaluados':refs}
            guardar_json(ev/f'LOSO/{contexto}.json',{'ID_ejecucion':args.ejecucion_id,**registro})
            estado['Pliegues'].append({'Fold':fold,'Sujeto':sub,'Estado':'COMPLETO','Pesos':meta['Pesos'],'Tiempo_s':registro['Tiempo_s']})
            guardar_json(ev/'Estado_LOSO.json',estado)
            listos=np.flatnonzero(np.isfinite(predicciones['Ensamble_ponderado']))
            parciales={n:p[listos] for n,p in predicciones.items()}
            guardar_csv(ev/'Predicciones_LOSO.csv',tabla_predicciones(desarrollo.iloc[listos],parciales,args.ejecucion_id))
            parcial=resumir(desarrollo.iloc[listos],parciales,modulos['tan'],'Val',False);parcial['ID_ejecucion']=args.ejecucion_id
            guardar_csv(resumen,parcial)
            print(f'COMPLETO {contexto}: {registro["Tiempo_s"]:.1f} s; pesos {dict(zip(ROLES,meta["Pesos"]))}',flush=True)
        for p in predicciones.values():validar_probabilidades(p,len(desarrollo))
        estado['Estado']='COMPLETO';guardar_json(ev/'Estado_LOSO.json',estado)
        base=resumir(desarrollo,predicciones,modulos['tan'],'Val');base['ID_ejecucion']=args.ejecucion_id
        guardar_csv(resumen,base)
        guardar_json(ev/'Complementariedad_LOSO.json',complementariedad(np.column_stack([predicciones[f'{r}_original'] for r in ROLES]),desarrollo.Ansiedad.to_numpy(),desarrollo.Sujeto.to_numpy(),ROLES))
        guardar_json(ev/'Seleccion_LOSO_resumen.json',{'ID_ejecucion':args.ejecucion_id,'Frecuencias_candidatos':dict(Counter(c['id'] for c in selecciones)),
                                                    'Pesos_medios':np.mean(pesos,axis=0).tolist(),'Roles':ROLES})
        contexto='Ajuste_final_32';final=ajustar_entrenamiento(desarrollo,banco,motor,contexto)
        # El conjunto externo se consulta solo después de fijar componentes,
        # calibradores, pesos, umbral y la referencia individual con desarrollo.
        testpred,modelos_finales=predecir_entrenamiento(desarrollo,externa,final,motor,contexto)
        guardar_json(ev/'Ensamble_final.json',{'ID_ejecucion':args.ejecucion_id,**final,'Modelos_base':modelos_finales})
        guardar_csv(ev/'Predicciones_Externas.csv',tabla_predicciones(externa,testpred,args.ejecucion_id))
        evaluacion=resumir(externa,testpred,modulos['tan'],'Test').drop(columns=['Estado','Umbral_fijo'])
        consolidado=base.merge(evaluacion,on='Modelo',validate='one_to_one');consolidado['Tiempo_total_s']=time.perf_counter()-inicio
        guardar_csv(resumen,consolidado)
        graficas(salida,predicciones,testpred,desarrollo.Ansiedad.to_numpy(),externa.Ansiedad.to_numpy(),pesos,final)
        diagnostico_calibracion(salida,ev,predicciones,testpred,desarrollo.Ansiedad.to_numpy(),externa.Ansiedad.to_numpy())
        guardar_json(ev/'Complementariedad_Externa.json',complementariedad(np.column_stack([testpred[f'{r}_original'] for r in ROLES]),externa.Ansiedad.to_numpy(),externa.Sujeto.to_numpy(),ROLES))
        config.update(Estado='COMPLETO',Tiempo_total_s=time.perf_counter()-inicio,Pesos_finales=dict(zip(ROLES,final['Pesos'])),
                      Componentes_finales=final['Seleccionados'],Referencia_individual_final=final['Referencia_mejor_original'])
        guardar_json(ev/'Configuracion.json',config)
        print(f'Resumen: {resumen}',flush=True)
        print(consolidado[['Modelo','Balanced_accuracy (Val)','Balanced_accuracy (Test)','ROC_AUC (Test)','Brier (Test)']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    except BaseException as exc:
        config.update(Estado='INTERRUMPIDO' if isinstance(exc,KeyboardInterrupt) else 'ERROR',Contexto=contexto,Error=f'{type(exc).__name__}: {exc}',Tiempo_total_s=time.perf_counter()-inicio)
        guardar_json(ev/'Configuracion.json',config)
        guardar_json(ev/'Error_ejecucion.json',{'ID_ejecucion':args.ejecucion_id,'Contexto':contexto,'Error':config['Error']})
        print(f'Se conservan resultados parciales y caché. Falló {contexto}: {config["Error"]}',flush=True)
        raise


if __name__=='__main__':
    principal()
