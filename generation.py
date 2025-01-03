import google.generativeai as genai

# Configure the API key
genai.configure(api_key="")

from google.generativeai.types import HarmCategory, HarmBlockThreshold

safe = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
}

def generate_response(context, retrieved_data, conversation_history):
    """
    Generate a response using a generative AI model in a Retrieval-Augmented Generation (RAG) setup.

    Parameters:
        context (str): The user's question or query.
        retrieved_data (str): Relevant information retrieved from the database.
        conversation_history (list): List of previous questions in the conversation.

    Returns:
        str: The response generated by the model.
    """
    # Select the appropriate model
    model = genai.GenerativeModel("gemini-1.0-pro")

    # Construct the prompt based on the availability of retrieved data
    if conversation_history:
        history = "\n".join(conversation_history)
        if retrieved_data:
            prompt = (
                "Bạn là một chuyên gia tư vấn luật pháp hàng đầu tại Việt Nam với kiến thức sâu rộng và kinh nghiệm thực tiễn. "
                "Dựa trên các thông tin được cung cấp dưới đây, hãy đưa ra câu trả lời chi tiết, dễ hiểu và có tính chuyên nghiệp cao để hỗ trợ người dùng:\n"
                "--- Thông tin tham khảo ---\n"
                f"{retrieved_data}\n\n"
                "--- Lịch sử cuộc trò chuyện ---\n"
                f"{history}\n\n"
                "--- Câu hỏi hiện tại của người dùng ---\n"
                f"{context}\n\n"
                "Hãy đảm bảo câu trả lời được trình bày một cách rõ ràng, chính xác, và phù hợp với ngữ cảnh pháp lý tại Việt Nam."
            )
        else:
            prompt = (
                "Bạn là một chuyên gia tư vấn luật pháp hàng đầu tại Việt Nam. "
                "Hiện tại, không có thông tin cụ thể nào để hỗ trợ trả lời câu hỏi của người dùng. "
                "Vui lòng yêu cầu người dùng cung cấp thêm thông tin chi tiết hoặc hướng dẫn họ tập trung vào các khía cạnh cụ thể hơn liên quan đến luật pháp Việt Nam."
            )
    else:
        if retrieved_data:
            prompt = (
                "Bạn là một chuyên gia tư vấn luật pháp hàng đầu tại Việt Nam với kiến thức sâu rộng và kinh nghiệm thực tiễn. "
                "Dựa trên các thông tin được cung cấp dưới đây, hãy đưa ra câu trả lời chi tiết, dễ hiểu và có tính chuyên nghiệp cao để hỗ trợ người dùng:\n"
                "--- Thông tin tham khảo ---\n"
                f"{retrieved_data}\n\n"
                "--- Câu hỏi hiện tại của người dùng ---\n"
                f"{context}\n\n"
                "Hãy đảm bảo câu trả lời được trình bày một cách rõ ràng, chính xác, và phù hợp với ngữ cảnh pháp lý tại Việt Nam."
            )
        else:
            prompt = (
                "Bạn là một chuyên gia tư vấn luật pháp hàng đầu tại Việt Nam. "
                "Hiện tại, không có thông tin cụ thể nào để hỗ trợ trả lời câu hỏi của người dùng. "
                "Vui lòng yêu cầu người dùng cung cấp thêm thông tin chi tiết hoặc hướng dẫn họ tập trung vào các khía cạnh cụ thể hơn liên quan đến luật pháp Việt Nam."
            )

    # Generate the response using the model
    
    try:
        response = model.generate_content(prompt, safety_settings=safe)
        if response and response.text:
            return response.text.strip()
        else:
            return "Lỗi: Không có phản hồi từ mô hình."
    except Exception as e:
        return f"Đã xảy ra lỗi khi gọi mô hình: {str(e)}"

