from rest_framework import serializers

class ResumeAnalysisSerializer(serializers.Serializer):
    resume = serializers.FileField()
    job_id = serializers.IntegerField()
