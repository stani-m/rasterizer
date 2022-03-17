use std::time::Instant;

use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

use rasterizer::clipper;
use rasterizer::presenter::WgpuPresenter;
use rasterizer::rasterizer::BresenhamLineRasterizer;
use rasterizer::{Color, ColorDepthBuffer, Pipeline, StripShapeAssembler, VertexOutput};
use crate::model::Model;

#[path = "../model.rs"]
mod model;

fn main() {
    let width = 800;
    let height = 600;
    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Lines example")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut framebuffer = ColorDepthBuffer::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, true);
    let mut pipeline = Pipeline::new(
        |vertex: glam::Vec4| VertexOutput {
            position: vertex,
            attributes: [],
        },
        StripShapeAssembler::new(),
        clipper::simple_line,
        BresenhamLineRasterizer::new(),
        |current, new| new <= current,
        |_| Color::new(1.0, 1.0, 1.0, 1.0),
    );
    let vertex_buffer = [
        glam::vec4(-0.5, -0.5, 0.0, 1.0),
        glam::vec4(-0.5, 0.5, 0.0, 1.0),
        glam::vec4(0.5, 0.5, 0.0, 1.0),
        glam::vec4(0.5, -0.5, 0.0, 1.0),
    ];
    let index_buffer = [0, 1, 2, 3, 0];

    let program_start = Instant::now();
    let mut last_frame_time = program_start;
    let mut last_second = 0;
    let mut frames = 0u32;

    event_loop.run_return(|event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(size) => {
                    let width = size.width;
                    let height = size.height;
                    framebuffer.resize(width, height);
                    presenter.resize(width, height);
                }
                _ => (),
            },
            Event::MainEventsCleared => {
                let current_frame_time = Instant::now();
                // let delta_time = current_frame_time - last_frame_time;
                let since_program_start = current_frame_time - program_start;

                let current_second = since_program_start.as_secs();
                if current_second != last_second {
                    println!("FPS: {frames}");
                    last_second = current_second;
                    frames = 0;
                }
                frames += 1;

                framebuffer.clear_color(Color::default());
                framebuffer.clear_depth(f32::INFINITY);
                pipeline.draw_indexed(&vertex_buffer, &index_buffer, &mut framebuffer);
                presenter.present(&framebuffer);

                last_frame_time = current_frame_time;
            }
            _ => (),
        }
    })
}
