import base64
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_avatar = "avatars/aisupport.png"
encoded_image_bot = get_base64_encoded_image(bot_avatar)
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="{img_source}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''.replace("{img_source}","data:image/png;base64," + encoded_image_bot)


user_avatar = "avatars/user.png"
encoded_image_user = get_base64_encoded_image(user_avatar)
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="{img_source}">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''.replace("{img_source}", "data:image/png;base64," + encoded_image_user)
