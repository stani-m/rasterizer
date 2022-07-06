use crate::FragmentInput;

pub fn passthrough<const A: usize, const V: usize>(
    shape: [FragmentInput<A>; V],
) -> Vec<[FragmentInput<A>; V]> {
    vec![shape]
}

pub fn simple<const A: usize, const V: usize>(
    shape: [FragmentInput<A>; V],
) -> Vec<[FragmentInput<A>; V]> {
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
