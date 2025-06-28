import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from django.conf import settings

class JobAPIClient:
    def __init__(self):
        self.base_url = settings.JOB_API_BASE_URL
        self.api_key = settings.JOB_API_KEY

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_job_details(self, job_id):
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(
                f"{self.base_url}/{job_id}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            json_data = response.json()

            if json_data.get("status") != "200":
                raise ValueError(f"Unexpected status code: {json_data.get('status')}")

            job_data = json_data.get("data", {}).get("job", {})
            if not job_data:
                raise ValueError("Job data not found in response")

            required_fields = {
                'title', 'job_type', 'salary', 'location',
                'job_status', 'description', 'requirement',
                'benefits', 'position'
            }
            missing = required_fields - job_data.keys()
            if missing:
                raise ValueError(f"Missing fields in job data: {', '.join(missing)}")

            return job_data

        except requests.RequestException as e:
            raise Exception(f"API request failed: {e}")
        except ValueError as ve:
            raise Exception(f"Invalid job data format: {ve}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
