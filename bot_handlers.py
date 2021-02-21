import gc
from nn import *

from urllib.parse import urljoin

import logging
import os

from aiogram import Bot,types,executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.webhook import SendMessage
from aiogram.utils.executor import start_webhook


BOT_TOKEN = os.environ['BOT_TOKEN']

WEBHOOK_HOST = os.environ['WEBHOOK_HOST_ADDR']
WEBHOOK_PATH = f'/webhook/{BOT_TOKEN}'
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)

WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = os.environ['PORT']

logging.basicConfig(level=logging.INFO)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


device = torch.device("cpu")

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

cnn = models.alexnet(pretrained=True).features.to(device).eval()




def image_loader(image_name):
    imsize = 196

    loader = transforms.Compose([
    transforms.Resize(imsize),  # нормируем размер изображения
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # превращаем в удобный формат

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor_save_rgbimage(tensor, filename):
    unloader = transforms.ToPILImage()
    img = tensor.clone()
    img = img.squeeze(0)
    img = unloader(img)
    img.save(filename)

flag = True
content_flag = False
style_flag = False


@dp.message_handler(commands=['start'])
async def test(message: types.Message):
    await message.answer(text='Hi! This bot is got two pics '
                              'and transfers the style of '
                              'the second pic to the first content pic. '
                              'To begin with, send a content image, please.')

@dp.message_handler(commands=['test'])
async def test(message: types.Message):
    """Test function."""
    await message.answer(text='It works!')

@dp.message_handler(content_types=['photo'])
async def photo_processing(message):

    global flag
    global content_flag
    global style_flag

    # The bot is waiting for a picture with content from the user.
    if flag:
        await message.photo[-1].download('content.jpg')
        await message.answer(text='OK! The content image is received. '
                                  'Now send a style image, please. '
                                  'You can use the /back command to '
                                  'change the content image.')
        flag = False
        content_flag = True  # Now the bot knows that the content image exists.

    # The bot is waiting for a picture with style from the user.
    else:
        await message.photo[-1].download('style.jpg')
        await message.answer(text='Wonderful! The style image is received.'
                                  'You can use the /back command to '
                                  'change the style image. '
                                  'Click /transfer to get the output pic.')
        flag = True
        style_flag = True  # Now the bot knows that the style image exists.

@dp.message_handler(commands=['back'])
async def photo_processing(message: types.Message):

    global flag
    global content_flag
    global style_flag

    # Let's make sure that there is something to cancel.

    if content_flag and not flag:
        flag = True
        content_flag = False
        await message.answer(text='Send a content image again, please.')
    else:
        flag = False
        style_flag = False
        await message.answer(text='Send a style image again, please.')

flag_final = True

@dp.message_handler(commands=['transfer'])
async def contin(message: types.Message):

    global flag_final
        
    # Let's make sure that the user has added both images.
    if not (content_flag * style_flag):  # Conjunction
        await message.answer(text="Send the pics again, please.")
        return
    if flag_final:
        await message.answer(text='Please wait a few minutes... The tranfer has begun!')
        flag_final = False
    
    style_img = image_loader('style.jpg')
    content_img = image_loader('content.jpg')
    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                     content_img, style_img, input_img)
    tensor_save_rgbimage(output.data[0], 'result.jpg')

    # Clear the RAM.
    with open('result.jpg', 'rb') as file:
        await message.answer_photo(file, caption='Great! This is the pic with the transferred style!')
    del style_img
    del content_img
    del input_img
    del output
    torch.cuda.empty_cache()


@dp.message_handler(lambda message: message.text in ("Great! This is the pic with the transferred style!"))
async def processing(message: types.Message):
    
    global flag_final
    
    if not flag_final:
        flag_final = True

async def on_startup(dp):
    # insert code here to run it after start]
    logging.warning('Hi!')
    await bot.set_webhook(WEBHOOK_URL)


async def on_shutdown(dp):
    logging.warning('Shutting down..')

    logging.warning('Bye!')


if __name__ == '__main__':
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=False,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT
    )
