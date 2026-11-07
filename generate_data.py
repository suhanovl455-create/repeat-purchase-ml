"""Generate synthetic transactions; the target is derived later, not generated directly."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def generate():
    rng = np.random.default_rng(2026)
    calendar = pd.date_range('2026-01-01', '2026-08-31')
    snapshot_dates = ['2026-05-01', '2026-06-01', '2026-07-01', '2026-08-01']
    snapshots, orders = [], []
    for uid in range(1, 1201):
        customer = f'C{uid:04}'
        # A latent purchasing rate is never exported as an input feature.
        rate = rng.beta(1.4, 4.0) * .12
        snapshot = snapshot_dates[(uid - 1) % 4]
        snapshots.append((customer, snapshot))
        for day in calendar[rng.random(len(calendar)) < rate]:
            orders.append((len(orders)+1, customer, day.date().isoformat(),
                           'cancelled' if rng.random() < .1 else 'paid',
                           int(rng.choice([99000, 199000, 499000, 999000]))))
    pd.DataFrame(snapshots, columns=['customer_id', 'as_of']).to_csv(ROOT/'data/snapshots.csv',index=False)
    pd.DataFrame(orders, columns=['order_id','customer_id','order_date','status','amount_kopecks']).to_csv(ROOT/'data/orders.csv',index=False)
    print(f'Generated {len(snapshots)} customer snapshots and {len(orders)} synthetic orders.')


if __name__ == '__main__': generate()
