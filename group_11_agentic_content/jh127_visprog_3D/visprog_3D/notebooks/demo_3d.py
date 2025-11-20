import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY_HERE'

from PIL import Image
from IPython.core.display import HTML
from functools import partial

from engine.utils import ProgramGenerator, ProgramInterpreter
from prompts.threedv import create_prompt

interpreter = ProgramInterpreter(dataset='threedv')

prompter = partial(create_prompt,method='all')
generator = ProgramGenerator(prompter=prompter)

from vggt.utils.load_fn import load_and_preprocess_images
images = load_and_preprocess_images(['../assets/threedv/room1.jpg',
                                     '../assets/threedv/room2.jpg'])
# images = load_and_preprocess_images(['../assets/threedv/table1.jpg',
#                                      '../assets/threedv/table2.jpg'])
# to PIL Image
image_list = [images[0].permute(1,2,0).numpy(), images[1].permute(1,2,0).numpy()]
image_list = [Image.fromarray((img * 255).astype('uint8')) for img in image_list]
init_state = dict(
    IMAGE_LIST=image_list
)
image = image_list[0]

# question = "Give me the size of this room in 3D (two images)?"
question = "Where is the plant in 3D (two images)?"
# question = "Remove the pear in 3D (two images)."
# question = "Compute distance between the pear and the wall in 3D (two images)."

prog,_ = generator.generate(dict(question=question))
print(prog)

result, prog_state, html_str = interpreter.execute(prog,init_state,inspect=True)

print("Final Result:", result)

# save html
with open('threedv_debug.html','w') as f:
    f.write(html_str)