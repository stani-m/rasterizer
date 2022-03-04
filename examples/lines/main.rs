use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use rasterizer::{Color, Framebuffer};
use rasterizer::presenter::WgpuPresenter;

fn main() {
    let width = 800;
    let height = 600;
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Lines example")
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    let mut framebuffer = Framebuffer::new(width, height);
    let mut presenter = WgpuPresenter::new(&window, width, height, true);

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
            framebuffer[[0, 0]] = Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            };
            presenter.present(&framebuffer);
        }
        _ => (),
    })
}
