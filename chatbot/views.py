from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .ai_chat import ask_gemini
from .serializers import ChatSerializer

class ChatAPIView(APIView):
    """
    POST { "message": "Your question" }
    → { "reply": "Bot answer" }
    """

    def post(self, request):
        serializer = ChatSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user_msg = serializer.validated_data['message'].strip()
        if not user_msg:
            return Response({'detail': 'Empty message'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            reply = ask_gemini(user_msg)
        except Exception as e:
            return Response(
                {'detail': f'Gemini error: {str(e)}'},
                status=status.HTTP_502_BAD_GATEWAY
            )

        return Response({'reply': reply})
