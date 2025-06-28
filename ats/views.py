import logging
import hashlib
import os
import tempfile
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from .models import ResumeAnalysis
from .serializers import ResumeAnalysisSerializer
from .services.job_api_client import JobAPIClient
from .utils.ai_processor import ResumeAnalyzer

logger = logging.getLogger(__name__)

class ResumeAnalysisView(APIView):
    parser_classes = (MultiPartParser,)
    
    def _validate_file(self, file):
        if not file.name.lower().endswith('.pdf'):
            raise ValueError("Only PDF files are allowed")
        if file.size > 5 * 1024 * 1024:
            raise ValueError("File size exceeds 5MB limit")

    def _get_job_data(self, job_id):
        try:
            return JobAPIClient().get_job_details(job_id)
        except Exception as e:
            logger.error(f"Job API failure: {str(e)}")
            raise RuntimeError(f"Could not retrieve job details: {str(e)}")

    def _process_resume(self, file_path, job_data, file_hash):
        analyzer = ResumeAnalyzer(skills_csv_path=settings.SKILLS_CSV_PATH)
        
        try:
            raw_text = analyzer.extract_text(file_path)
            analysis = analyzer.analyze(raw_text, job_data)
            
            # Extract applicant info from analysis
            personal_info = analysis.get('cv_data', {}).get('personal_info', {})
            
            # Check for existing analysis
            existing = ResumeAnalysis.objects.filter(
                file_hash=file_hash,
                job_data=job_data  # Direct JSON comparison
            ).first()
            
            if existing:
                return existing

            # Save file
            fs = FileSystemStorage()
            with open(file_path, 'rb') as f:
                filename = fs.save(
                    f"resumes/{file_hash}.pdf",
                    ContentFile(f.read())
                )

            return ResumeAnalysis.objects.create(
                resume=filename,
                job_data=job_data,
                analysis_result=analysis,
                file_hash=file_hash,
                applicant_name=personal_info.get('name', ''),
                applicant_email=personal_info.get('email', '')
            )
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    def post(self, request):
        serializer = ResumeAnalysisSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            resume_file = serializer.validated_data['resume']
            job_id = serializer.validated_data['job_id']
            
            self._validate_file(resume_file)
            job_data = self._get_job_data(job_id)
            
            # Generate file hash
            file_content = resume_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            
            # Check for existing analysis
            existing = ResumeAnalysis.objects.filter(
                file_hash=file_hash,
                job_data=job_data
            ).first()
            
            if existing:
                return Response({
                    'message': 'Using cached analysis',
                    'analysis_id': existing.id,
                    'job_data': existing.job_data,
                    'result': existing.analysis_result
                }, status=status.HTTP_200_OK)

            # Process file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            try:
                record = self._process_resume(tmp_path, job_data, file_hash)
                return Response({
                    'analysis_id': record.id,
                    'job_data': record.job_data,
                    'result': record.analysis_result
                }, status=status.HTTP_201_CREATED)
            finally:
                os.remove(tmp_path)

        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except RuntimeError as e:
            return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
        except Exception as e:
            logger.exception("Analysis failed")
            return Response(
                {"error": "Internal server error"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )