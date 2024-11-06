# Generated by Django 5.1.2 on 2024-11-05 12:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0006_delete_feedbackdata'),
    ]

    operations = [
        migrations.AddField(
            model_name='patientrecord',
            name='strain_type',
            field=models.CharField(choices=[('HIV-1', 'HIV-1'), ('HIV-2', 'HIV-2')], default='HIV-1', max_length=10, null=True),
        ),
    ]