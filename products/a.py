from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from products.form import ProductForm
from products.recommendation import *
from .models import Order, Product, Transaction, Cart
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
import numpy as np
import logging
from django.shortcuts import render, redirect
from .models import Product
import seaborn as sns
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse
from io import BytesIO
import base64
from .models import Cart, Transaction, Product
from django.shortcuts import render, redirect
from products.recommendation import *
from .models import Product
import logging


# Set up logging
logger = logging.getLogger(__name__)
import logging

# Set up logging
logger = logging.getLogger(__name__)




import logging
import requests
from bs4 import BeautifulSoup
from .models import Product
from requests.exceptions import RequestException
logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
import logging

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product
import logging

logger = logging.getLogger(__name__)

import logging
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Product


logger = logging.getLogger(__name__)


@login_required
def add_to_cart_view(request, product_id):
    """
    Handle adding a product to the cart. If the product is already in the cart, update the quantity.
    """
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
    """
    Handle updating the quantity of an item in the cart.
    """
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
    """
    Process the user's cart and create an order.

    Parameters:
        request: The HTTP request.

    Returns:
        redirect: Redirect to the order summary page after checkout.
    """
    cart_items = Cart.objects.filter(user=request.user)
    if not cart_items:
        return redirect('products:cart_view')  # Redirect to cart if there are no items in the cart

    total_price = sum(item.total_price() for item in cart_items)

    # Create an order for the user
    order = Order.objects.create(user=request.user, total_price=total_price)

    # Create transactions for each cart item
    for item in cart_items:
        transaction = Transaction.objects.create(
            user=request.user,
            product=item.product,
            quantity=item.quantity,
            price=item.product.price,
            total_amount=item.total_price(),
        )
        order.transactions.add(transaction)  # Corrected this line to use `transactions.add`

    # Clear the cart after the order is created
    cart_items.delete()

    # Redirect to a page where the user can view their order (you can customize this)
    return redirect('products:order_summary', order_id=order.id)



@login_required
def cart_view(request):
    """
    Displays the user's cart items and the total price.
    """
    cart_items = Cart.objects.filter(user=request.user)
    total = 0  # Initialize total to 0
    
    # Calculate the total price for the cart
    for item in cart_items:
        total += item.product.price * item.quantity
    
    return render(request, 'cart.html', {'cart_items': cart_items, 'total': total})

@login_required
def remove_from_cart(request, cart_item_id):
    """
    Remove an item from the cart.
    """
    cart_item = get_object_or_404(Cart, id=cart_item_id, user=request.user)
    cart_item.delete()
    return redirect('products:cart_view')  # Redirect back to the cart view after removal


@login_required
def order_summary(request, order_id):
    """
    Display the order summary after the user completes checkout.

    Parameters:
        request: The HTTP request.
        order_id (int): The ID of the order to display.

    Returns:
        render: The order summary page.
    """
    order = get_object_or_404(Order, id=order_id)
    return render(request, 'order_summary.html', {'order': order})



# Initialize logger
logger = logging.getLogger(__name__)

def scrape_product(url):
    logger.info(f"Scraping products from URL: {url}")
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 ...'})
        
        # Log the raw HTML content to check if we're getting the product data
        if response.status_code == 200:
            logger.info(response.text[:1000])  # Log first 1000 characters to check the response structure
            soup = BeautifulSoup(response.content, 'html.parser')

            # Log the prettified version of the HTML to ensure correct structure
            logger.info(soup.prettify()[:1000])  # Log first 1000 characters of prettified HTML
            
            products = soup.find_all('div', {'class': 'search-result-gridview-item-wrapper'})
            logger.info(f"Found {len(products)} products.")  # Log the number of products found
            
            if not products:
                logger.warning(f"No products found on the page: {url}")
                return
            
            # Continue scraping and saving logic
            for product in products:
                name = product.find('span', {'class': 'prod-ProductTitle'}).text.strip() if product.find('span', {'class': 'prod-ProductTitle'}) else None
                price = product.find('span', {'class': 'price-main'}).text.strip() if product.find('span', {'class': 'price-main'}) else None
                rating = product.find('span', {'class': 'stars-reviews'}).text.strip() if product.find('span', {'class': 'stars-reviews'}) else None
                image_url = 'https://www.walmart.com' + product.find('a')['href'] if product.find('a') else None

                if name and price and rating and image_url:
                    Product.objects.update_or_create(
                        name=name,
                        defaults={'price': price, 'rating': rating, 'product_link': image_url}
                    )
                    logger.info(f"Saved product: {name}, {price}, {rating}, {image_url}")
                else:
                    logger.warning(f"Skipping product due to missing data: {name}, {price}, {rating}, {image_url}")
        else:
            logger.error(f"Failed to retrieve the page. Status code: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error occurred: {e}")

def product_list(request):
    """
    Displays a list of products along with recommended products based on Collaborative Filtering.
    Scrapes new products from an external site if not already scraped, and loads recommendations.
    """
    if not request.user.is_authenticated:
        return redirect('users:login')  # Redirect to login if the user is not authenticated

    # List of external product URLs to scrape from (currently, Walmart is the primary source)
    external_product_urls = [
        'https://www.walmart.com/search/?query=laptops',  # Walmart search URL for laptops
        'https://www.walmart.com/search/?query=smartphones&cat_id=0',  # Walmart search URL for smartphones
    ]

    # Scrape products from each external URL
    try:
        for url in external_product_urls:
            logger.info(f"Scraping products from URL: {url}")
            scrape_product(url)  # Call the scraping function to save products into the database

        # After scraping, fetch the updated list of products from the database
        products = Product.objects.all().order_by('-id')  # Show most recently added products first
        logger.info(f"Fetched updated product list: {products.count()} products found.")

    except Exception as e:
        logger.error(f"Error scraping products: {e}")
        products = Product.objects.all().order_by('-id')  # Fallback to existing products if scraping fails

    # Load recommendations based on collaborative filtering (if implemented)
    try:
        # Train the collaborative filtering model and get recommended product IDs
        recommendations, product_ids = train_collaborative_filtering(request.user)

        # Fetch the recommended products from the database
        recommended_products = Product.objects.filter(id__in=product_ids)

        # Pass both products and recommended products to the template context
        context = {
            'products': products,
            'recommended_products': recommended_products,
        }

        logger.info("Loaded recommendations based on collaborative filtering.")

    except Exception as e:
        logger.error(f"Error loading recommendations: {e}")
        context = {
            'products': products,
            'recommended_products': []  # Fallback if recommendations fail
        }

    # Render the page with both products and recommendations
    return render(request, 'product_list.html', context)


# Product Create (Create new product)
def product_create(request):
    """
    Handle the creation of a new product.

    Parameters:
        request: The HTTP request.

    Returns:
        render: The product creation form page.
    """
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
    """
    Display the product details and show recommended products.

    Parameters:
        request: The HTTP request.
        pk (int): The product ID.

    Returns:
        render: The product detail page with recommended products.
    """
    product = get_object_or_404(Product, pk=pk)

    if request.method == 'POST' and 'add_to_cart' in request.POST:
        quantity = int(request.POST.get('quantity', 1))
        cart_item, created = Cart.objects.get_or_create(user=request.user, product=product)

        if not created:
            cart_item.quantity += quantity
            cart_item.save()

        return redirect('product_detail', pk=product.pk)

    # Get recommended products (excluding current one)
    recommended_products = Product.objects.exclude(id=product.id)[:5]

    return render(request, 'product_detail.html', {
        'product': product,
        'recommended_products': recommended_products,
    })





# Product Edit (Edit existing product)
def product_edit(request, pk):
    """
    Handle editing an existing product.

    Parameters:
        request: The HTTP request.
        pk (int): The ID of the product to edit.

    Returns:
        render: The product editing form page.
    """
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        form = ProductForm(request.POST, request.FILES, instance=product)
        if form.is_valid():
            form.save()
            return redirect('products:product_list')
    else:
        form = ProductForm(instance=product)
    return render(request, 'product_form.html', {'form': form})
# Product Delete (Delete a product)
def product_delete(request, pk):
    """
    Handle the deletion of a product.

    Parameters:
        request: The HTTP request.
        pk (int): The ID of the product to delete.

    Returns:
        redirect: Redirect to the product list after deletion.
    """
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        product.delete()
        return redirect('products:product_list')
    return render(request, 'delete_product.html', {'product': product})




def plot_user_item_matrix(user_item_matrix):
    """
    Visualize the user-item matrix using a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_item_matrix, cmap='Blues', cbar=True, annot=False, xticklabels=10, yticklabels=10)
    plt.title("User-Item Interaction Heatmap")
    plt.xlabel("Product")
    plt.ylabel("User ID")

    # Save to BytesIO and encode as base64 string
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def plot_similarity_matrix(similarity_matrix, item_ids):
    """
    Visualize the similarity matrix using a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarity_matrix, cmap='viridis', annot=False, xticklabels=item_ids[:10], yticklabels=item_ids[:10])
    plt.title("Item Similarity Matrix (Top 10 Products)")
    plt.xlabel("Products")
    plt.ylabel("Products")

    # Save to BytesIO and encode as base64 string
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def visualize_recommendations(request):
    """
    View to render the dashboard page with visualizations and recommendations.
    """
    # Get the user-item matrix
    user_item_matrix = get_user_item_matrix()

    # Build SVD model and get similarity matrix
    similarity_matrix, item_ids = build_svd_model()

    # Generate association rules
    rules = generate_association_rules(min_support=0.01, min_threshold=1.0)

    # Get recommendations for a specific user
    user_id = 1  # For example, user_id=1
    recommendations = get_recommendations(user_id, user_item_matrix, top_n=5)

    # Plot visualizations
    user_item_matrix_plot = plot_user_item_matrix(user_item_matrix)
    similarity_matrix_plot = plot_similarity_matrix(similarity_matrix, item_ids)

    # Pass the context to the template
    context = {
        'user_item_matrix_plot': user_item_matrix_plot,
        'similarity_matrix_plot': similarity_matrix_plot,
        'recommendations': recommendations,
    }

    return render(request, 'dashboard.html', context)
