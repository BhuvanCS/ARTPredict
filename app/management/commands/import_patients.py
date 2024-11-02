import csv
from django.core.management.base import BaseCommand
from app.models import PatientRecord

class Command(BaseCommand):
    help = 'Imports patient data from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help="Path to the CSV file containing patient data")

    def handle(self, *args, **kwargs):
        csv_file_path = kwargs['csv_file']
        
        with open(csv_file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                PatientRecord.objects.create(
                    patient_id=row['Patient_ID'],
                    viral_load=float(row['Viral_Load']),
                    cd4_count=int(row['CD4_Count']),
                    adherence_level=float(row['Adherence_Level']),
                    treatment_response=row['Treatment_Response']
                )
                self.stdout.write(self.style.SUCCESS(f"Added patient {row['Patient_ID']}"))

        self.stdout.write(self.style.SUCCESS("All patients imported successfully."))
