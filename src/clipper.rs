use crate::FragmentInput;

pub fn passthrough<const A: usize, const S: usize>(
    shape: [FragmentInput<A>; S],
) -> Vec<[FragmentInput<A>; S]> {
    vec![shape]
}

pub fn simple<const A: usize, const S: usize>(
    shape: [FragmentInput<A>; S],
) -> Vec<[FragmentInput<A>; S]> {
    #[rustfmt::skip]
    let out = {
        shape.iter().all(|vertex| vertex.position.x > vertex.position.w)
            || shape.iter().all(|vertex| vertex.position.x < -vertex.position.w)
            || shape.iter().all(|vertex| vertex.position.y > vertex.position.w)
            || shape.iter().all(|vertex| vertex.position.y < -vertex.position.w)
            || shape.iter().all(|vertex| vertex.position.z > vertex.position.w)
            || shape.iter().all(|vertex| vertex.position.z < 0.0)
    };
    if !out {
        vec![shape]
    } else {
        Vec::new()
    }
}
