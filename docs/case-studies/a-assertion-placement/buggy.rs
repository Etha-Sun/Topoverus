use vstd::prelude::*;

verus! {

fn count_less_than(numbers: Vec<u32>, threshold: u32) -> (count: u32)
    requires
        true,
    ensures
        count == numbers.len() as u32,
{
    let mut shrink = numbers.clone();
    let mut grow: Vec<u32> = Vec::new();
    while shrink.len() > 0
        invariant
            grow.len() + shrink.len() == numbers.len(),
        decreases shrink.len(),
    {
        let i = shrink.pop().unwrap();
        assert(grow.len() + shrink.len() == numbers.len());
        grow.push(i);
    }
    assert(numbers.len() == grow.len());
    grow.len() as u32
}

fn main() {
}

} // verus!