from django.db import models
from django.contrib.auth.models import User


class Product(models.Model):
    ELECTRONICS = 'electronics'
    FURNITURE = 'furniture'
    CLOTHING = 'clothing'
    BOOKS = 'books'
    TOYS = 'toys'
    GROCERIES = 'groceries'
    ACCESSORIES='Accessories'

    CATEGORY_CHOICES = [
        (ELECTRONICS, 'Electronics'),
        (FURNITURE, 'Furniture'),
        (CLOTHING, 'Clothing'),
        (BOOKS, 'Books'),
        (TOYS, 'Toys'),
        (GROCERIES, 'Groceries'),
        (ACCESSORIES,'Accessories')
    ]
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    rating = models.CharField(max_length=50)
    category = models.CharField(
        max_length=20,
        choices=CATEGORY_CHOICES,
    )
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.IntegerField(default=0)  # Number of items in stock
    image = models.ImageField(upload_to='products/', null=True, blank=True)
    product_link = models.URLField()
    rating = models.CharField(max_length=50)
    is_recommended = models.BooleanField(default=False)
    def __str__(self):
        return self.name

class Cart(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    rating = models.CharField(max_length=50)
    product_link = models.URLField()
    image_url = models.URLField()
    added_at = models.DateTimeField(auto_now_add=True)

    def total_price(self):
        return self.quantity * self.product.price

    def __str__(self):
        return f"Cart Item: {self.product.name} - {self.quantity}"
class CartItem(models.Model):
    cart = models.ForeignKey(Cart, related_name='items', on_delete=models.CASCADE)  # Related to Cart model
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)


class ProductLike(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    liked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Like {self.user.username} liked {self.product.name}"

class Transaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    purchased_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transaction {self.id} - {self.product.name}"


class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    date_created = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='Pending')
    transactions = models.ManyToManyField(Transaction, related_name='orders')

    def __str__(self):
        return f"Order {self.id} - {self.user.username}"
    
class OrderTransaction(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE)
    
    def __str__(self):
        return f"Order {self.order.id} - Transaction {self.transaction.id}"


class UserInteraction(models.Model):
  
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    interaction_type = models.CharField(max_length=50)  
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} {self.interaction_type} {self.product.name} at {self.timestamp}"

class RecommendationLog(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    recommended_products = models.JSONField()  # Store a list of recommended product IDs
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Recommendations for {self.user.username} at {self.timestamp}"

