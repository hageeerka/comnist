import asyncio
from use_module import *

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message


bot = Bot(token='7191286169:AAGhrpQALBMxdfAA2VoBfOYu7sEiXLxuTKQ')
dp = Dispatcher()

@dp.message(CommandStart())
async def start_message(message: Message):
    await message.answer('Привет, добро пожаловать в чат-бот по распознаванию текста\n'
                         '\n'
                         'Отправьте нам фото с рукописным текстом')

@dp.message(F.photo)
async def photo_handler(message: Message) -> str:
    id_photo = message.photo[-1].file_id
    file_name = f"photos/{id_photo}.jpg"
    await message.bot.download(file=id_photo, destination=file_name)
    await message.answer('Спасибо за фото, началась обработка информации')
    result = start_module(file_name)
    await message.answer('Вот ваш текст:\n'
                         f'{result}\n')
    return id_photo

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Выход из программы')