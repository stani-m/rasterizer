use std::time::Instant;

use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

use model::Model;
use rasterizer::presenter::WgpuPresenter;
use rasterizer::rasterizer::{CullFace, EdgeFunctionRasterizer};
use rasterizer::{clipper, Color, ColorDepthBuffer, ListShapeAssembler, Pipeline, FragmentInput};

#[path = "../model.rs"]
mod model;

fn main() {
    let width = 800;
    let height = 600;
    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Triangles example")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut framebuffer = ColorDepthBuffer::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, true);
    let mut pipeline = Pipeline::new(
        |(position, uv): (glam::Vec3, glam::Vec2), &transform| FragmentInput {
            position: transform * glam::Vec4::from((position, 1.0)),
            attributes: uv.to_array(),
        },
        ListShapeAssembler::new(),
        clipper::simple,
        EdgeFunctionRasterizer::new(Some(CullFace::Ccw)),
        |current, new| new <= current,
        |fragment_input, _| {
            let pos = (glam::Vec2::from(fragment_input.attributes) * 24.0).as_ivec2() % 2;
            if pos.x == pos.y {
                Color::new(0.0, 1.0, 1.0, 1.0)
            } else {
                Color::new(1.0, 0.0, 1.0, 1.0)
            }
        },
    );

    let (document, buffers, _) = gltf::import("examples/assets/Cube.gltf")
        .expect("glTF import failed, file probably not found, make sure to run examples from crate root directory");
    let mut cube = Model::from(&document, &buffers[0]);
    let uvs: Vec<glam::Vec2> =
        cube.load_vertex_attribute(gltf::Semantic::TexCoords(0), &buffers[0]);
    let cube_vertex_uv_buffer = cube
        .vertex_buffer()
        .to_vec()
        .into_iter()
        .zip(uvs)
        .collect::<Vec<_>>();

    let zoom = glam::Mat4::from_scale(glam::vec3(0.5, 0.5, 0.5));
    let view = glam::Mat4::look_at_lh(
        glam::vec3(1.0, 2.0, 3.0),
        glam::vec3(0.0, 0.0, 0.0),
        glam::vec3(0.0, 1.0, 0.0),
    );
    let projection = glam::Mat4::perspective_lh(
        45_f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        100.0,
    );
    let mut camera = projection * view * zoom;

    let program_start = Instant::now();
    let mut last_frame_time = program_start;
    let mut last_second = 0;
    let mut frames = 0u32;

    event_loop.run_return(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                let width = size.width;
                let height = size.height;
                framebuffer.resize(width, height);
                presenter.resize(width, height);
                let projection = glam::Mat4::perspective_lh(
                    45_f32.to_radians(),
                    width as f32 / height as f32,
                    0.1,
                    100.0,
                );
                camera = projection * view * zoom;
            }
            _ => (),
        },
        Event::MainEventsCleared => {
            let current_frame_time = Instant::now();
            let delta_time = current_frame_time - last_frame_time;
            let since_program_start = current_frame_time - program_start;

            let current_second = since_program_start.as_secs();
            if current_second != last_second {
                println!("FPS: {frames}");
                last_second = current_second;
                frames = 0;
            }
            frames += 1;

            let rotate = glam::Quat::from_axis_angle(
                glam::vec3(0.0, 1.0, 0.0),
                -delta_time.as_secs_f32() * 0.3,
            );
            cube.rotation *= rotate;
            let transform = camera * cube.model_matrix();

            framebuffer.clear_color(Color::default());
            framebuffer.clear_depth(f32::INFINITY);
            pipeline.draw_indexed(
                &cube_vertex_uv_buffer,
                &cube.index_buffer(),
                &transform,
                &mut framebuffer,
            );

            presenter.present(&framebuffer);

            last_frame_time = current_frame_time;
        }
        _ => (),
    });
}
