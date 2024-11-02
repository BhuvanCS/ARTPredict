import re
from django.core.management.base import BaseCommand
from app.models import PatientRecord

class Command(BaseCommand):
    help = 'Update PatientRecord with sequence data from a FASTA file'

    def add_arguments(self, parser):
        parser.add_argument('fasta_file', type=str, help='Path to the FASTA file containing sequence data')

    def handle(self, *args, **kwargs):
        fasta_file = kwargs['fasta_file']
        
        try:
            with open(fasta_file, 'r') as file:
                patient_id = None
                sequence_data = ""
                
                for line in file:
                    line = line.strip()
                    
                    # Check for header line starting with ">"
                    if line.startswith(">"):
                        if patient_id and sequence_data:
                            self.save_sequence_data(patient_id, sequence_data)
                            
                        # Patient_ID from header (format >P_1 | ...)
                        header_parts = line[1:].split("|")
                        patient_id = header_parts[0].strip()
                        sequence_data = ""  
                    
                    else:
                        sequence_data += line
                
                if patient_id and sequence_data:
                    self.save_sequence_data(patient_id, sequence_data)
            
            self.stdout.write(self.style.SUCCESS('Successfully updated PatientRecord with sequence data.'))
        
        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"File {fasta_file} not found."))

    def save_sequence_data(self, patient_id, sequence_data):
        try:
            patient_record = PatientRecord.objects.get(patient_id=patient_id)
            patient_record.sequence_data = sequence_data
            patient_record.save()
            self.stdout.write(f"Updated sequence data for {patient_id}")
        
        except PatientRecord.DoesNotExist:
            self.stdout.write(self.style.WARNING(f"No PatientRecord found for {patient_id}"))
