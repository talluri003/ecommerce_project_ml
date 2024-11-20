from django import forms
from .models import Product

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'price', 'description', 'image','category'] 

        widgets = {
            'description': forms.Textarea(attrs={'rows': 4, 'cols': 40}),
        }
