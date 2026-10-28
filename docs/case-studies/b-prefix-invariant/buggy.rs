use vstd::math::{max, min};
use vstd::prelude::*;

verus! {

pub fn min_i32(x: i32, y: i32) -> (res: i32)
    ensures
        res == min(x as int, y as int),
{
    if x < y { x } else { y }
}

pub fn max_i32(x: i32, y: i32) -> (res: i32)
    ensures
        res == max(x as int, y as int),
{
    if x < y { y } else { x }
}

pub fn max_profit(prices: Vec<i32>) -> (res: i32)
    requires
        1 <= prices.len() <= 100000,
        forall|i: int| 0 <= i < prices.len() ==> 0 <= #[trigger] prices[i] <= 10000,
    ensures
        forall|j: int, k: int| 0 <= j < k < prices.len() ==> prices[k] - prices[j] <= res,
        res > 0 ==> exists|j: int, k: int|
            0 <= j < k < prices.len() && prices[k] - prices[j] == res,
{
    let mut max_pro = 0;
    let mut min_price = i32::MAX;

    // Has the key proof context: price bounds, the prefix lower-bound fact
    // for min_price, and a witness that min_price comes from the seen prefix.
    // But the top-level optimality invariant is still wrong: it quantifies over
    // the full array instead of only the processed prefix `i`.
    for i in 0..prices.len()
        invariant
            i == 0 ==> min_price == i32::MAX,
            forall|j: int| 0 <= j < prices.len() ==> 0 <= #[trigger] prices[j] <= 10000,
            forall|j: int| 0 <= j < i ==> #[trigger] prices[j] >= min_price,
            i > 0 ==> exists|j: int| 0 <= j < i && #[trigger] prices[j] == min_price,
            max_pro > 0 ==> exists|j: int, k: int|
                0 <= j < k < i && prices[k] - prices[j] == max_pro,
            forall|j: int, k: int| 0 <= j < k < prices.len() ==> prices[k] - prices[j] <= max_pro,
    {
        min_price = min_i32(min_price, prices[i]);
        max_pro = max_i32(max_pro, prices[i] - min_price);
    }

    max_pro
}

} // verus!
fn main() {}
