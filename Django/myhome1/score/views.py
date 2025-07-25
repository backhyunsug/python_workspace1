from django.shortcuts import render, redirect 
from django.http import HttpResponse, HttpRequest 
# Create your views here.

def index(request):
    return HttpResponse("score")

def list(request):
    return HttpResponse("score_list")

def view(request, id):
    return HttpResponse("score_list")

def write(request):
    return HttpResponse("score_list")

def save(request):
    return HttpResponse("score_list")
