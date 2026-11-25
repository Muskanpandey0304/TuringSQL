# Data quality log

Every dropped or changed example, with the reason and the count. Filled in during data preparation
(PROJECT_PLAN.md section 3.2).

| Date | Dataset | Issue | Decision | Rows affected |
|---|---|---|---|---|
| 2026-11-24 | Olist marts | `fact_orders.customer_state` differs from the customer's home state in `dim_customer.state` (the customer used a different address for some orders) | Questions about a customer's **region** take both state and region from `dim_customer`, so the two always agree. Questions that only need the order's state use `fact_orders.customer_state`. | 43 orders, 39 customers |
| 2026-11-24 | Olist marts | `is_late` compares timestamps, and the estimated date is stored as midnight. So 1,291 of 7,822 "late" orders in the analysis window arrived **on** the estimated day (average review 4.03, the same as on-time orders). By calendar date the late rate is 6.8%, not 8.1%. | The benchmark keeps the warehouse flag as the definition (E30 now asks for orders "marked as delivered late"). Reported to the Olist project as a definition issue. | 1,291 orders |
| 2026-11-24 | Olist marts | Orders stop in October 2018, but the analysis window ends in August 2018, so Sep and Oct 2018 look like a collapse | Month-over-month questions are limited to the window or to complete years | 2018-09: 16 orders, 2018-10: 4 orders |
| 2026-11-24 | Spider train | Duplicate (db_id, question) pairs | Dropped | 25 |
| 2026-11-24 | Spider train | Gold SQL errors in SQLite | Dropped | 3 |
| 2026-11-24 | Spider train / dev | Gold SQL returns more than 10,000 rows (above the evaluator's row cap) | Dropped | 6 / 2 |
| 2026-11-24 | Spider train | Gold SQL returns an empty result (empty tables in the database) | Kept | 1,606 |
| 2026-11-24 | Spider train | Prompt + SQL longer than 2,048 tokens (large schemas) | Dropped from SFT | 242 |
