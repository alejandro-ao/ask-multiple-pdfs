css = '''
<style>
.chat-message {
    padding: 1rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    max-width: 100%;
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 15%;
}
.chat-message .avatar img {
  max-width: 48px;
  max-height: 48px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 85%;
  padding: 0 1rem;
  color: #fff;
  font-size: 0.95rem;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://static.wixstatic.com/media/0e9f63_efefe52050da4ddbb4d24bee75da80cf~mv2.png/v1/fill/w_448,h_110,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/DraftyAI%20Logo_AI%20for%20Immigration%20Lawyers.png" style="max-height: 48px; max-width: 48px; object-fit: contain; background-color: white; padding: 2px;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://openclipart.org/download/247319/abstract-user-flat-3.svg" style="max-height: 48px; max-width: 48px;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
