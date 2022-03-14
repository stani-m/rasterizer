use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use rasterizer::presenter::WgpuPresenter;
use rasterizer::{line_bresenham, Color, ColorBuffer, ListShapeAssembpler, Pipeline, VertexOutput};

fn main() {
    let width = 800;
    let height = 600;
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Lines example")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut framebuffer = ColorBuffer::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, true);
    let mut pipeline = Pipeline::new(
        |vertex: glam::Vec4| VertexOutput {
            position: vertex,
            attributes: [],
        },
        ListShapeAssembpler::new(),
        line_bresenham,
        |_| Color::new(1.0, 1.0, 1.0, 1.0),
    );

    event_loop.run(move |event, _, control_flow| match event {
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
            framebuffer.clear(Color::default());
            let vertex_buffer = [
                glam::vec4(-1.0, -1.0, 0.0, 1.0),
                glam::vec4(1.0, 1.0, 2.0, 2.0),
            ];
            pipeline.draw(&vertex_buffer, &mut framebuffer);
            presenter.present(&framebuffer);
        }
        _ => (),
    })
}
