use crate::VertexOutput;

pub fn passthrough<const A: usize, const S: usize>(
    shape: [VertexOutput<A>; S],
) -> Vec<[VertexOutput<A>; S]> {
    vec![shape]
}

pub fn simple_line<const A: usize>(shape: [VertexOutput<A>; 2]) -> Vec<[VertexOutput<A>; 2]> {
    let [from, to] = &shape;
    let from = &from.position;
    let to = &to.position;
    let from_w = from.w;
    let to_w = to.w;
    let xy_out =
        (0..2).any(|i| from[i] > from_w && to[i] > to_w || from[i] < -from_w && to[i] < -to_w);
    let z_out = from.z > from_w && to.z > to_w || from.z < 0.0 && to.z < 0.0;
    if !(xy_out || z_out) {
        vec![shape]
    } else {
        Vec::new()
    }
}
