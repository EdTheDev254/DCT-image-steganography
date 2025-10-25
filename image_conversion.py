from PIL import Image

# Open, convert, and save in one chained command
Image.open('rick-has-a-secret.png').convert('RGB').save('rick_40q.jpg', quality=40)

