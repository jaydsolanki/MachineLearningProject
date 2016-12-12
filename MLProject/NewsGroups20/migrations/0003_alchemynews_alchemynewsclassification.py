# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-12-12 01:46
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('NewsGroups20', '0002_livenews'),
    ]

    operations = [
        migrations.CreateModel(
            name='AlchemyNews',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.CharField(blank=True, max_length=1024, null=True)),
                ('category', models.CharField(blank=True, max_length=25, null=True)),
            ],
            options={
                'db_table': 'alchemy_news',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='AlchemyNewsClassification',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('category', models.IntegerField(blank=True, null=True)),
                ('score', models.FloatField(blank=True, null=True)),
                ('algorithm', models.CharField(blank=True, max_length=25, null=True)),
            ],
            options={
                'db_table': 'alchemy_news_classification',
                'managed': False,
            },
        ),
    ]