import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from picamera2 import Picamera2
import io
import base64
from PIL import Image, ImageDraw
import numpy as np
import atexit

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Img(id='image'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000, # in milliseconds
        n_intervals=0
    ),
    html.Div(id='rgb-value'),
    html.Button('Calibrate', id='calibrate-button', n_clicks=0),
    dcc.Input(id='target-r', type='number', placeholder='Target R value'),
    dcc.Input(id='target-g', type='number', placeholder='Target G value'),
    dcc.Input(id='target-b', type='number', placeholder='Target B value')
])

# Create a Picamera2 object and start the camera
picam2 = Picamera2()
picam2.start(show_preview=False)

@app.callback(
    [Output('image', 'src'), Output('rgb-value', 'children')],
    [Input('interval-component', 'n_intervals'), Input('calibrate-button', 'n_clicks')],
    [State('target-r', 'value'), State('target-g', 'value'), State('target-b', 'value')]
)
def update_image(n, n_clicks, target_r, target_g, target_b):
    # Capture an image as a PIL Image object
    image = picam2.capture_image("main")

    # Convert the image to RGB mode if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Take a sample of pixels from the center of the screen and output the average RGB value
    width, height = image.size
    center_x, center_y = width // 2, height // 2
    sample_size = 10

    sample_box = (center_x - sample_size // 2, center_y - sample_size // 2,
                  center_x + sample_size // 2, center_y + sample_size // 2)
    sample_image = image.crop(sample_box)
    sample_array = np.array(sample_image)
    avg_rgb = np.mean(sample_array, axis=(0, 1))

    # Draw a box on the image to represent the sample area
    draw = ImageDraw.Draw(image)
    draw.rectangle(sample_box, outline=(0,0,0), width=3)

    # Calibrate the camera if the calibrate button was clicked
    if n_clicks > 0 and all(v is not None for v in [target_r, target_g, target_b]):
        target_rgb = np.array([target_r, target_g, target_b])
        calibration_offset = target_rgb - avg_rgb
        
        #metadata = picam2.capture_metadata()
        #current_gains = metadata["ColourGains"]
        #gains = [current_gains[0]+calibration_offset[0],current_gains[1]+calibration_offset[1],current_gains[2]+calibration_offset[2]]
        #picam2.set_controls({"ColourGains": gains})

        # Reset the calibrate button clicks to prevent repeated calibration
        n_clicks = 0

    # Save the image data to a BytesIO object
    image_data = io.BytesIO()
    image.save(image_data, format='JPEG')

    # Encode the image data as a base64 string
    encoded_image = base64.b64encode(image_data.getvalue()).decode('ascii')

    # Return the data URL for the image and the average RGB value
    return f'data:image/jpeg;base64,{encoded_image}', f'Average RGB value: {avg_rgb}'

def stop_camera():
    picam2.stop()

atexit.register(stop_camera)

if __name__ == '__main__':
    app.run_server()
