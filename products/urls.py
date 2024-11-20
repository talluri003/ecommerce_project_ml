
from django.urls import path
from . import views
app_name = 'products' 

urlpatterns = [
    path('products', views.product_list, name='product_list'),
    path('product/<int:pk>/', views.product_detail, name='product_detail'),
    path('add_product/', views.product_create, name='product_create'),
    path('product/edit/<int:pk>/', views.product_edit, name='product_edit'),
    path('product/delete/<int:pk>/', views.product_delete, name='product_delete'),
    path('add-to-cart/<int:product_id>/', views.add_to_cart_view, name='add_to_cart'),
    path('cart/', views.cart_view, name='cart_view'),
    path('checkout/', views.checkout, name='checkout'),
    path('checkout/order_summary/<int:order_id>/', views.order_summary, name='order_summary'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('cart/update/<int:cart_item_id>/', views.update_cart_item, name='update_cart_item'),
    path('cart/remove/<int:cart_item_id>/', views.remove_from_cart, name='remove_from_cart'),
    path('get_cart_count/', views.get_cart_count, name='get_cart_count'),

]
