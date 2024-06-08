import telebot
import requests
from io import BytesIO

TELEGRAM_TOKEN = "token"
bot = telebot.TeleBot(TELEGRAM_TOKEN)

FASTAPI_URL = "http://127.0.0.1:8000/predict"


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправьте мне изображение, и я скажу вам, что это за еда.")


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    file = requests.get(f'https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_info.file_path}')
    img = BytesIO(file.content)

    files = {'file': img}
    response = requests.post(FASTAPI_URL, files=files)

    if response.status_code == 200:
        result = response.json()
        bot.reply_to(message, f"Это {result['class']}")
    else:
        bot.reply_to(message, "Произошла ошибка при распознавании изображения.")


if __name__ == '__main__':
    bot.polling(none_stop=True)
