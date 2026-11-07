import unittest
import pandas as pd
from features import FEATURES, build_dataset, temporal_split


def orders(rows):
    return pd.DataFrame(rows,columns=['order_id','customer_id','order_date','status','amount_kopecks'])


class FeatureTests(unittest.TestCase):
    def test_exact_time_boundaries(self):
        raw=orders([(1,'A','2026-01-31','paid',10000),
                    (2,'A','2026-01-30','paid',90000),
                    (3,'A','2026-04-30','paid',20000),
                    (4,'A','2026-05-01','paid',50000),
                    (5,'A','2026-05-31','paid',90000)])
        snap=pd.DataFrame([('A','2026-05-01')],columns=['customer_id','as_of'])
        row=build_dataset(raw,snap).iloc[0]
        self.assertEqual(row['orders_90d'],2)
        self.assertEqual(row['spend_90d_rub'],300)
        self.assertEqual(row['recency_days'],1)
        self.assertEqual(row['returned_30d'],1)
        # May 31 is exactly the exclusive target end and must not count.
        row=build_dataset(raw.loc[raw['order_id']!=4],snap).iloc[0]
        self.assertEqual(row['returned_30d'],0)

    def test_future_never_changes_features(self):
        past=orders([(1,'A','2026-04-30','paid',10000)])
        future=orders([(2,'A','2026-05-10','paid',999999)])
        snap=pd.DataFrame([('A','2026-05-01')],columns=['customer_id','as_of'])
        before=build_dataset(past,snap);after=build_dataset(pd.concat([past,future]),snap)
        pd.testing.assert_frame_equal(before[FEATURES],after[FEATURES])
        self.assertEqual(before.iloc[0]['returned_30d'],0)
        self.assertEqual(after.iloc[0]['returned_30d'],1)

    def test_cancellation_and_no_previous_purchase(self):
        raw=orders([(1,'A','2026-04-30','paid',10000),
                    (2,'A','2026-05-10','cancelled',99999),
                    (3,'B','2026-05-10','paid',20000)])
        snap=pd.DataFrame([('A','2026-05-01'),('B','2026-05-01')],columns=['customer_id','as_of'])
        result=build_dataset(raw,snap)
        self.assertEqual(result['customer_id'].tolist(),['A'])
        self.assertEqual(result.iloc[0]['returned_30d'],0)

    def test_incomplete_future_is_excluded(self):
        raw=orders([(1,'A','2026-08-01','paid',10000)])
        snap=pd.DataFrame([('A','2026-08-10')],columns=['customer_id','as_of'])
        self.assertTrue(build_dataset(raw,snap,observed_until='2026-09-01').empty)

    def test_split(self):
        data=pd.DataFrame([('A','2026-06-01','2026-07-01'),('B','2026-07-01','2026-07-31')],
                          columns=['customer_id','as_of','label_end_exclusive'])
        train,test=temporal_split(data)
        self.assertEqual(len(train),1);self.assertEqual(len(test),1)
        data.loc[0,'label_end_exclusive']='2026-07-02'
        with self.assertRaises(ValueError):temporal_split(data)

    def test_duplicate_snapshot_or_order_rejected(self):
        raw=orders([(1,'A','2026-04-30','paid',10000)])
        snap=pd.DataFrame([('A','2026-05-01')],columns=['customer_id','as_of'])
        with self.assertRaises(ValueError):build_dataset(pd.concat([raw,raw]),snap)
        with self.assertRaises(ValueError):build_dataset(raw,pd.concat([snap,snap]))


if __name__=='__main__':unittest.main()
