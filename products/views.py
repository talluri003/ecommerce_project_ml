# Standard Library Imports
import logging
import io
import base64
import numpy as np
import requests
from io import BytesIO
from requests.exceptions import RequestException

# Third-Party Imports
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup

# Django Imports
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required

# App-Specific Imports
from products.form import ProductForm
from products.recommendation import *
from .models import Order, Product, Transaction, Cart

# Logger
logger = logging.getLogger(__name__)
@login_required
def add_to_cart_view(request, product_id):

    product = get_object_or_404(Product, id=product_id)
    quantity = int(request.POST.get('quantity', 1))  # Default quantity is 1 if not specified

    # Get or create the cart item for the user and product
    cart_item, created = Cart.objects.get_or_create(user=request.user, product=product)

    if not created:
        cart_item.quantity += quantity  # If the item is already in the cart, increase the quantity
        cart_item.save()

    return redirect('products:cart_view')  



@login_required
def update_cart_item(request, cart_item_id):

    cart_item = get_object_or_404(Cart, id=cart_item_id, user=request.user)
    
    if request.method == 'POST':
        # Get the new quantity from the form
        new_quantity = int(request.POST.get('quantity', 1))

        # Ensure quantity is at least 1
        if new_quantity >= 1:
            cart_item.quantity = new_quantity
            cart_item.save()
    
    return redirect('products:cart_view')  

@login_required
def checkout(request):

    cart_items = Cart.objects.filter(user=request.user)
    if not cart_items:
        return redirect('products:cart_view')  

    total_price = sum(item.total_price() for item in cart_items)

    order = Order.objects.create(user=request.user, total_price=total_price)

    for item in cart_items:
        transaction = Transaction.objects.create(
            user=request.user,
            product=item.product,
            quantity=item.quantity,
            price=item.product.price,
            total_amount=item.total_price(),
        )
        order.transactions.add(transaction) 

    cart_items.delete()

    return redirect('products:order_summary', order_id=order.id)

def get_cart_count(request):
    cart =Cart.objects.filter(user=request.user).first()  
    cart_item_count = cart.items.count() if cart else 0  
    return render(request, 'product_list.html', {'cart_item_count': cart_item_count})

@login_required
def cart_view(request):
    cart_items = Cart.objects.filter(user=request.user)
    total = 0  # Initialize total to 0
    
    # Calculate the total price for the cart
    for item in cart_items:
        total += item.product.price * item.quantity
    
    return render(request, 'cart.html', {'cart_items': cart_items, 'total': total})

@login_required
def remove_from_cart(request, cart_item_id):
    cart_item = get_object_or_404(Cart, id=cart_item_id, user=request.user)
    cart_item.delete()
    return redirect('products:cart_view')  # Redirect back to the cart view after removal


@login_required
def order_summary(request, order_id):
    order = get_object_or_404(Order, id=order_id)
    return render(request, 'order_summary.html', {'order': order})

@login_required

def product_create(request):
 
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('products:product_list')  # Redirect to the product list after saving
    else:
        form = ProductForm()

    return render(request, 'product_form.html', {'form': form})





@login_required
def product_detail(request, pk):
    product = get_object_or_404(Product, pk=pk)

    if request.method == 'POST' and 'add_to_cart' in request.POST:
        quantity = int(request.POST.get('quantity', 1))
        cart_item, created = Cart.objects.get_or_create(user=request.user, product=product)

        if not created:
            cart_item.quantity += quantity
            cart_item.save()

        return redirect('product_detail', pk=product.pk)

    recommended_products = Product.objects.exclude(id=product.id)[:5]

    return render(request, 'product_detail.html', {
        'product': product,
        'recommended_products': recommended_products,
    })





def product_edit(request, pk):
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES, instance=product)
        if form.is_valid():
            form.save()
            return redirect('products:product_list')
    else:
        form = ProductForm(instance=product)
    return render(request, 'product_form.html', {'form': form})



def product_delete(request, pk):
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        product.delete()
        return redirect('products:product_list')
    return render(request, 'delete_product.html', {'product': product})



logger = logging.getLogger(__name__)

def product_list(request):
    if not request.user.is_authenticated:
        return redirect('users:login')  # Redirect to login page if user is not authenticated

    try:
        user_id = request.user.id

        selected_category = request.GET.get('category', '')
        search_query = request.GET.get('search', '')

        products = Product.objects.all()

        if selected_category:
            products = products.filter(category=selected_category)
        if search_query:
            products = products.filter(name__icontains=search_query)

        products = products.order_by('-id')

        paginator = Paginator(products, 15) 
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        # Get combined recommendations (KNN or popular products)
        recommended_products = get_combined_recommendations(user_id, top_n=5)

        # Render the template and pass the products and recommendations
        context = {
            'products': page_obj,  # Paginated product list
            'recommended_products': recommended_products,
            'selected_category': selected_category,  # Pass the selected category for filtering
            'categories': Product.CATEGORY_CHOICES,  # Pass category choices to the template for filtering
            'search_query': search_query,  # Pass the search query for the search filter
        }

    except Exception as e:
        logger.error(f"Error occurred while fetching products or recommendations: {e}", exc_info=True)

        context = {
            'products': Product.objects.all(),  
            'recommended_products': [],
            'error_message': 'An error occurred while fetching the products and recommendations. Please try again later.',
        }

    return render(request, 'product_list.html', context)

def on_cart_update(user_id, product_id, interaction_type='cart', top_n=5):
    try:
        logger.info(f"User {user_id} updated their {interaction_type} with product {product_id}.")
        updated_recommendations = get_combined_recommendations(user_id, top_n=top_n)
        return updated_recommendations
    except Exception as e:
        logger.error(f"Error updating cart recommendations for user {user_id}: {e}")
        return []  # Return empty list if an error occurs

@login_required
def dashboard(request):
    try:
        user_id = request.user.id

        # Get KNN-based recommendations (or fallback to popular products)
        recommended_products = get_combined_recommendations(user_id, top_n=5)

        # Create or update the user-item matrix and generate the heatmap
        user_item_matrix, user_item_matrix_plot = create_user_item_matrix()

        # Generate a plot of popular products
        popular_products_plot = plot_popular_products(top_n=5)

        # Evaluate the models (KNN, Association Rules, Popular Products)
        knn_evaluation = evaluate_knn_recommendations(user_id, top_n=5)
        association_evaluation = evaluate_association_rule_recommendations(user_id, top_n=5)
        popular_products_evaluation = evaluate_popular_products_recommendations(user_id, top_n=5)

        # Visualize the evaluation results
        plot_url = visualize_evaluation_results(
            knn_evaluation, 
            association_evaluation, 
            popular_products_evaluation
        )

        # Check for updates in the user's cart or transactions
        if 'update_cart' in request.POST:
            product_id = request.POST.get('product_id')
            interaction_type = request.POST.get('interaction_type', 'cart')  # Default to 'cart' if not provided
            # Trigger the real-time update after cart interaction (add/update/remove)
            updated_recommendations = on_cart_update(user_id, product_id, interaction_type=interaction_type, top_n=5)
            # Get updated recommendations (fallback to popular products if necessary)
            recommended_products = get_combined_recommendations(user_id, top_n=5)

        #  Prepare context for rendering
        context = {
            'recommended_products': recommended_products,
            'user_item_matrix': user_item_matrix,  # Pass the user-item matrix if needed
            'user_item_matrix_plot': user_item_matrix_plot,
            'popular_products_plot': popular_products_plot,
            'plot_url': plot_url,  # Add the evaluation plot URL
            'user_id': user_id,  # Pass the user_id to the template for user-specific content
        }

        return render(request, 'dashboard.html', context)

    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return render(request, 'dashboard.html', {'error': 'An error occurred while generating recommendations.'})
def visualize_evaluation_results(knn_evaluation, association_evaluation, popular_products_evaluation):
    """
    Visualizes the evaluation results (Precision, Recall, F1) for KNN, Association Rules, and Popular Products.
    """
    try:
        # Prepare data for plotting
        models = ['KNN', 'Association Rules', 'Popular Products']
        precision_scores = [knn_evaluation['precision'], association_evaluation['precision'], popular_products_evaluation['precision']]
        recall_scores = [knn_evaluation['recall'], association_evaluation['recall'], popular_products_evaluation['recall']]
        f1_scores = [knn_evaluation['f1'], association_evaluation['f1'], popular_products_evaluation['f1']]

        # Create a DataFrame for plotting
        eval_df = pd.DataFrame({
            'Model': models,
            'Precision': precision_scores,
            'Recall': recall_scores,
            'F1-Score': f1_scores
        })

        # Plot the evaluation metrics
        plt.figure(figsize=(10, 6))
        eval_df.set_index('Model').plot(kind='bar', stacked=False, width=0.8, color=['#4CAF50', '#FF9800', '#2196F3'])
        plt.title('Evaluation of Recommendation Models')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.xticks(rotation=0)
        plt.legend(title='Metrics', loc='upper left')

        # Save the plot to a BytesIO buffer and encode it as base64 for rendering in templates
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return plot_url  # Return the base64-encoded string for rendering in templates

    except Exception as e:
        logger.error(f"Error occurred while visualizing evaluation results: {e}")
        return None  # Return None if error occurs
