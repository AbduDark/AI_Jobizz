from django.db import models
import os
import uuid

def resume_upload_path(instance, filename):
    ext = os.path.splitext(filename)[1]
    unique_id = uuid.uuid4().hex
    return os.path.join('resumes', f'resume_{unique_id}{ext}')

class ResumeAnalysis(models.Model):
    resume = models.FileField(upload_to=resume_upload_path)
    job_data = models.JSONField(null=True)  # Store full job data
    analysis_result = models.JSONField()
    applicant_name = models.CharField(max_length=255, blank=True)
    applicant_email = models.EmailField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    file_hash = models.CharField(max_length=32, null=True)  # MD5 hash

    class Meta:
        verbose_name_plural = "Resume Analyses"
        ordering = ['-created_at']
        unique_together = ['file_hash', 'job_data']

    def delete(self, *args, **kwargs):
        if self.resume:
            if os.path.isfile(self.resume.path):
                os.remove(self.resume.path)
        super().delete(*args, **kwargs)