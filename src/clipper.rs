use crate::UnshadedFragment;

pub fn passthrough<const A: usize, const N: usize>(
    shape: [UnshadedFragment<A>; N],
) -> Vec<[UnshadedFragment<A>; N]> {
    vec![shape]
}

pub fn simple<const A: usize, const N: usize>(
    shape: [UnshadedFragment<A>; N],
) -> Vec<[UnshadedFragment<A>; N]> {
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

const INSIDE: u8 = 0b000000;
const LEFT: u8 = 0b000001;
const RIGHT: u8 = 0b000010;
const BOTTOM: u8 = 0b000100;
const TOP: u8 = 0b001000;
const NEAR: u8 = 0b010000;
const FAR: u8 = 0b100000;

fn compute_out_code(vertex: &glam::Vec4) -> u8 {
    let mut out_code = INSIDE;

    if vertex.x < -vertex.w {
        out_code |= LEFT;
    } else if vertex.x > vertex.w {
        out_code |= RIGHT;
    }

    if vertex.y < -vertex.w {
        out_code |= BOTTOM;
    } else if vertex.y > vertex.w {
        out_code |= TOP;
    }

    if vertex.z < 0.0 {
        out_code |= NEAR;
    } else if vertex.z > vertex.w {
        out_code |= FAR;
    }

    out_code
}

pub fn sutherland_hodgman_near_far<const A: usize>(
    triangle: [UnshadedFragment<A>; 3],
) -> Vec<[UnshadedFragment<A>; 3]> {
    let out_code_0 = compute_out_code(&triangle[0].position);
    let out_code_1 = compute_out_code(&triangle[1].position);
    let out_code_2 = compute_out_code(&triangle[2].position);

    if out_code_0 & out_code_1 & out_code_2 != INSIDE {
        return Vec::new();
    }

    let out_code = out_code_0 | out_code_1 | out_code_2;
    if out_code & (NEAR | FAR) != 0 {
        let polygon = clip_polygon(&triangle, out_code);

        let mut res = Vec::with_capacity((polygon.len() - 2) * 3);

        for vertices in polygon[1..].windows(2) {
            if vertices.len() >= 2 {
                res.push([polygon[0], vertices[0], vertices[1]]);
            }
        }

        return res;
    }

    vec![triangle]
}

fn clip_polygon<const A: usize>(
    polygon: &[UnshadedFragment<A>],
    out_code: u8,
) -> Vec<UnshadedFragment<A>> {
    let mut polygon = polygon.to_vec();

    // if out_code & LEFT != 0 {
    //     res = clip_by_plane(&mut res, |v| v.x >= -v.w, clip_x_left, |v| v.x = -v.w);
    // }
    // if out_code & RIGHT != 0 {
    //     res = clip_by_plane(&mut res, |v| v.x <= v.w, clip_x_right, |v| v.x = v.w);
    // }
    // if out_code & BOTTOM != 0 {
    //     res = clip_by_plane(&mut res, |v| v.y >= -v.w, clip_y_bottom, |v| v.y = -v.w);
    // }
    // if out_code & TOP != 0 {
    //     res = clip_by_plane(&mut res, |v| v.y <= v.w, clip_y_top, |v| v.y = v.w);
    // }
    if out_code & FAR != 0 {
        polygon = clip_by_plane(&mut polygon, |v| v.z <= v.w, clip_z_far, |v| v.z = v.w);
    }
    if out_code & NEAR != 0 {
        polygon = clip_by_plane(&mut polygon, |v| v.z >= 0.0, clip_z_near, |v| v.z = 0.0);
    }

    for fragment in &polygon {
        if fragment.position.w <= 0.0 {
            return Vec::new();
        }
    }

    polygon
}

fn clip_by_plane<const A: usize>(
    polygon: &[UnshadedFragment<A>],
    is_inside: fn(&glam::Vec4) -> bool,
    compute_t: fn(&glam::Vec4, &glam::Vec4) -> f32,
    clip: fn(&mut glam::Vec4),
) -> Vec<UnshadedFragment<A>> {
    let mut res = Vec::with_capacity(polygon.len());
    for i in 0..polygon.len() {
        let v0 = &polygon[i];
        let v1 = &polygon[(i + 1) % polygon.len()];

        if is_inside(&v0.position) {
            if is_inside(&v1.position) {
                res.push(*v1);
            } else {
                let t = compute_t(&v0.position, &v1.position);
                let mut new_fragment = mix_fragments(v0, v1, t);
                clip(&mut new_fragment.position);
                res.push(new_fragment);
            }
        } else {
            if is_inside(&v1.position) {
                let t = compute_t(&v0.position, &v1.position);
                let mut new_fragment = mix_fragments(v0, v1, t);
                clip(&mut new_fragment.position);
                res.push(new_fragment);
                res.push(*v1);
            }
        }
    }
    res
}

fn mix_fragments<const A: usize>(
    a: &UnshadedFragment<A>,
    b: &UnshadedFragment<A>,
    t: f32,
) -> UnshadedFragment<A> {
    *a + (*b - *a) * t
}

// fn clip_x_left(a: &glam::Vec4, b: &glam::Vec4) -> f32 {
//     (a.x + a.w) / ((a.x + a.w) - (b.x + b.w))
// }
//
// fn clip_x_right(a: &glam::Vec4, b: &glam::Vec4) -> f32 {
//     (a.x - a.w) / ((a.x - a.w) - (b.x - b.w))
// }
//
// fn clip_y_bottom(a: &glam::Vec4, b: &glam::Vec4) -> f32 {
//     (a.y + a.w) / ((a.y + a.w) - (b.y + b.w))
// }
//
// fn clip_y_top(a: &glam::Vec4, b: &glam::Vec4) -> f32 {
//     (a.y - a.w) / ((a.y - a.w) - (b.y + b.w))
// }

fn clip_z_far(a: &glam::Vec4, b: &glam::Vec4) -> f32 {
    (a.z - a.w) / ((a.z - a.w) - (b.z - b.w))
}

fn clip_z_near(a: &glam::Vec4, b: &glam::Vec4) -> f32 {
    a.z / (a.z - b.z)
}
