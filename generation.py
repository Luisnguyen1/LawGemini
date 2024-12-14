import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyCBK_dl3hPaRb-o2by0-fzkkvAXoOPVWsw")

def generate_response(context, retrieved_data):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    if retrieved_data:
        prompt = f"Dựa trên thông tin sau, hãy trả lời câu hỏi của người dùng:\n\n{retrieved_data}\n\nCâu hỏi: {context}"
    else:
        prompt = "Không tìm thấy thông tin liên quan. Vui lòng hỏi các câu hỏi liên quan đến luật pháp Việt Nam."

    response = model.generate_content(prompt)
    if response:
        return f"{response.text}"
    else:
        return "Lỗi: Không có phản hồi"
