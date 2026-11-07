"""Fit one preselected model and evaluate once on a later time period."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)
from features import FEATURES, build_dataset, temporal_split

ROOT = Path(__file__).resolve().parent


def evaluate(y, probabilities):
    predicted = (probabilities >= .5).astype(int)
    return {'accuracy':accuracy_score(y,predicted),
            'precision':precision_score(y,predicted,zero_division=0),
            'recall':recall_score(y,predicted,zero_division=0),
            'f1':f1_score(y,predicted,zero_division=0),
            'roc_auc':roc_auc_score(y,probabilities),
            'average_precision':average_precision_score(y,probabilities),
            'brier_score':brier_score_loss(y,probabilities),
            'confusion_matrix':confusion_matrix(y,predicted,labels=[0,1]).tolist()}


def main():
    out=ROOT/'results';out.mkdir(exist_ok=True)
    dataset=build_dataset(pd.read_csv(ROOT/'data/orders.csv'),pd.read_csv(ROOT/'data/snapshots.csv'))
    train,test=temporal_split(dataset)
    X_train,y_train=train[FEATURES],train['returned_30d']
    X_test,y_test=test[FEATURES],test['returned_30d']
    if y_train.nunique()!=2 or y_test.nunique()!=2:
        raise ValueError('Both classes are required in train and test for these metrics')
    # Scaling is learned only on train, inside the pipeline.
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
    baseline=DummyClassifier(strategy='prior')
    model.fit(X_train,y_train);baseline.fit(X_train,y_train)
    probability=model.predict_proba(X_test)[:,1]
    baseline_probability=baseline.predict_proba(X_test)[:,1]
    results={'logistic_regression':evaluate(y_test,probability),
             'baseline':evaluate(y_test,baseline_probability)}
    report={'data':'synthetic','features':FEATURES,'threshold':.5,
            'train_rows':len(train),'test_rows':len(test),
            'train_return_rate':float(y_train.mean()),'test_return_rate':float(y_test.mean()),
            'train_dates':sorted(train['as_of'].unique().tolist()),
            'test_dates':sorted(test['as_of'].unique().tolist()),'metrics':results}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    dataset.to_csv(out/'features_and_labels.csv',index=False)
    predictions=test.copy()
    predictions['probability']=probability
    predictions['baseline_probability']=baseline_probability
    predictions['prediction']=(probability>=.5).astype(int)
    predictions.to_csv(out/'test_predictions.csv',index=False)
    # Report standardized coefficients; these are associations, not causal effects.
    pd.DataFrame({'feature':FEATURES,'standardized_coefficient':model[-1].coef_[0]}).to_csv(out/'coefficients.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    RocCurveDisplay.from_predictions(y_test,probability,name='Logistic regression',ax=axes[0])
    RocCurveDisplay.from_predictions(y_test,baseline_probability,name='Baseline',ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test,probability,name='Logistic regression',ax=axes[1])
    PrecisionRecallDisplay.from_predictions(y_test,baseline_probability,name='Baseline',ax=axes[1])
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4.5))
    ConfusionMatrixDisplay.from_predictions(y_test,(probability>=.5).astype(int),
        display_labels=['No return','Return'],ax=ax,colorbar=False,cmap='Blues')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    lines=['# Результаты эксперимента','',
        'Данные синтетические. Модель и порог 0,5 выбраны заранее; настройки по тесту не подбирались.',
        '',f'Обучение: {len(train)} покупателей, май и июнь. Проверка: {len(test)} покупателей, июль и август.',
        f'Доля повторных покупок в тесте: {y_test.mean():.1%}.',
        '', '| Метрика | Логистическая регрессия | Базовый прогноз |', '|---|---:|---:|']
    for metric in ['accuracy','precision','recall','f1','roc_auc','average_precision','brier_score']:
        lines.append(f"| {metric} | {results['logistic_regression'][metric]:.3f} | {results['baseline'][metric]:.3f} |")
    tn,fp,fn,tp=confusion_matrix(y_test,(probability>=.5).astype(int),labels=[0,1]).ravel()
    lines+=['',f'При пороге 0,5: верно предсказано возвращение {tp} клиентов; пропущено {fn} возвращений; ложных прогнозов возвращения {fp}; верно предсказано отсутствие возвращения {tn}.',
        '', 'На этом наборе модель лучше baseline по ROC-AUC, average precision, Brier score и accuracy, но уступает ему по recall и F1 при пороге 0,5. Baseline всегда отвечает «вернётся» и потому не пропускает возвращения, одновременно ошибаясь на всех невозвратившихся клиентах. Улучшение ранжирования не гарантирует улучшения каждой пороговой метрики.', '', 'ROC-AUC оценивает ранжирование. Average precision оценивает качество ранжирования положительного класса; ориентир для константного прогноза — доля положительного класса. Brier score — средняя квадратичная ошибка вероятностей (меньше лучше). Accuracy, precision, recall и F1 здесь зависят от выбранного порога.',
        '', 'Baseline назначает всем вероятность, равную доле возвращений в обучении. При пороге 0,5 это превращается в прогноз одного класса. Различие метрик на этом наборе не доказывает качество на реальных клиентах; доверительные интервалы не рассчитывались.',
        '', '![ROC и PR кривые](curves.png)', '', '![Матрица ошибок](confusion_matrix.png)', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
