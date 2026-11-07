"""Build features from past orders and labels from a separate future window."""
import pandas as pd

FEATURES = ['recency_days', 'orders_90d', 'spend_90d_rub']


def build_dataset(orders, snapshots, observed_until='2026-09-01'):
    orders = orders.copy()
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    observed_until = pd.Timestamp(observed_until)  # exclusive end of available data
    if orders['order_id'].duplicated().any():
        raise ValueError('order_id must be unique')
    if snapshots['customer_id'].duplicated().any():
        raise ValueError('Each customer must have only one snapshot')
    groups = {uid: group for uid, group in orders.groupby('customer_id')}
    rows = []
    for row in snapshots.itertuples(index=False):
        as_of = pd.Timestamp(row.as_of)
        end = as_of + pd.Timedelta(days=30)
        if end > observed_until:
            continue  # incomplete target window: do not label this as non-return
        own = groups.get(row.customer_id, orders.iloc[:0])
        paid = own.loc[own['status'] == 'paid']
        past = paid.loc[(paid['order_date'] >= as_of-pd.Timedelta(days=90)) & (paid['order_date'] < as_of)]
        if past.empty:
            continue  # predict repeat purchase only for recent paying customers
        future = paid.loc[(paid['order_date'] >= as_of) & (paid['order_date'] < end)]
        rows.append({'customer_id': row.customer_id, 'as_of': as_of.date().isoformat(),
                     'label_end_exclusive': end.date().isoformat(),
                     'recency_days': (as_of-past['order_date'].max()).days,
                     'orders_90d': len(past),
                     'spend_90d_rub': int(past['amount_kopecks'].sum())/100,
                     'returned_30d': int(not future.empty)})
    return pd.DataFrame(rows, columns=['customer_id','as_of','label_end_exclusive']+FEATURES+['returned_30d'])


def temporal_split(dataset):
    train = dataset.loc[dataset['as_of'] < '2026-07-01'].copy()
    test = dataset.loc[dataset['as_of'] >= '2026-07-01'].copy()
    if train.empty or test.empty:
        raise ValueError('Both train and test must contain rows')
    if set(train['customer_id']) & set(test['customer_id']):
        raise ValueError('Customer overlap across train and test')
    if train['label_end_exclusive'].max() > test['as_of'].min():
        raise ValueError('Training labels would not yet be available at first test date')
    return train, test
