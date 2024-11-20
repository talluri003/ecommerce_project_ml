# Standard Library Imports
import logging
import io
import base64

# Third-Party Imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD  # For dimensionality reduction
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Django Imports
from django.db.models import Count

# App-Specific Imports
from products.models import Cart, Transaction, Product

# Set matplotlib to use a non-GUI backend (for server-side rendering)
matplotlib.use('Agg')

# Logger
logger = logging.getLogger(__name__)

def create_user_item_matrix():
    try:
        transactions = Transaction.objects.all().values('user_id', 'product_id')
        transaction_df = pd.DataFrame(transactions)

        if transaction_df.empty:
            logger.warning("No transaction data found.")
            return pd.DataFrame(), None
        
        # Assign weights based on transaction type (purchases = 1)
        transaction_df['interaction'] = 1
        
        cart_items = Cart.objects.all().values('user_id', 'product_id')
        cart_df = pd.DataFrame(cart_items)

        if cart_df.empty:
            logger.warning("No cart data found.")
            return pd.DataFrame(), None
        
        # Assign weights for cart items (e.g., 0.5, you could tweak this further)
        cart_df['interaction'] = 0.5
        
        # Merge cart and transaction data into a single dataframe
        df = pd.concat([transaction_df[['user_id', 'product_id', 'interaction']], 
                        cart_df[['user_id', 'product_id', 'interaction']]], ignore_index=True)
        
        # Pivot the data to create a user-item interaction matrix
        user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='interaction', aggfunc='sum', fill_value=0)

        if user_item_matrix.empty:
            logger.warning("User-item matrix is empty. No recommendations available.")
            return pd.DataFrame(), None
        
        return user_item_matrix, None

    except Exception as e:
        logger.error(f"Error occurred while creating user-item matrix: {e}")
        return pd.DataFrame(), None

def plot_user_item_matrix(user_item_matrix):
    try:
        if user_item_matrix.empty:
            logger.warning("User-item matrix is empty. Cannot generate heatmap.")
            return None  # Return None if the matrix is empty

        plt.figure(figsize=(12, 8))
        sns.heatmap(user_item_matrix, cmap="YlGnBu", annot=False, cbar=True)
        plt.title("User-Item Interaction Matrix Heatmap")
        plt.xlabel("Product ID")
        plt.ylabel("User ID")

        # Save the plot to a BytesIO buffer and encode it as base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return plot_url  # Return the base64 encoded string for rendering in the template

    except Exception as e:
        logger.error(f"Error occurred while plotting user-item matrix: {e}")
        return None

def get_knn_recommendations(user_id, top_n=5):
    try:
        user_item_matrix, _ = create_user_item_matrix()

        if user_item_matrix.empty:
            logger.warning(f"User-item matrix is empty for user {user_id}. Cannot generate KNN recommendations.")
            return []

        recommended_product_ids = knn_recommend_products(user_id, user_item_matrix, top_n)

        if not recommended_product_ids:
            logger.warning(f"No recommendations generated for user {user_id} using KNN.")

        return recommended_product_ids

    except Exception as e:
        logger.error(f"Error occurred while generating KNN recommendations for user {user_id}: {e}")
        return []


def knn_recommend_products(user_id, user_item_matrix, top_n=5):
    try:
        if user_id not in user_item_matrix.index:
            logger.warning(f"User {user_id} not found in the user-item matrix. Cannot generate KNN recommendations.")
            return []

        # Dynamically set n_components to be the number of features in the user-item matrix
        n_features = user_item_matrix.shape[1]  # Get the number of features (columns)
        n_components = min(20, n_features)  # Ensure n_components is not greater than the number of features

        # Perform dimensionality reduction using TruncatedSVD (for sparse matrix)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_matrix = svd.fit_transform(user_item_matrix)

        # Now, use the reduced matrix for KNN
        user_vector = reduced_matrix[user_item_matrix.index.get_loc(user_id)].reshape(1, -1)

        # Check the number of available samples (users) in the dataset
        n_samples = reduced_matrix.shape[0]
        actual_neighbors = min(top_n + 1, n_samples)  # Don't ask for more neighbors than available

        # Perform KNN to find similar users (neighbors)
        knn = NearestNeighbors(n_neighbors=actual_neighbors, metric='cosine')
        knn.fit(reduced_matrix)

        distances, indices = knn.kneighbors(user_vector)

        # Get the recommended product IDs from the nearest neighbors (excluding the user themselves)
        recommended_product_ids = user_item_matrix.columns[indices.flatten()[1:]].tolist()

        logger.info(f"KNN recommendations for user {user_id}: {recommended_product_ids}")

        return recommended_product_ids

    except Exception as e:
        logger.error(f"Error occurred while generating KNN recommendations for user {user_id}: {e}")
        return []

def get_combined_recommendations(user_id, top_n=5):
    try:
        # Try KNN-based recommendations first
        knn_recommendations = get_knn_recommendations(user_id, top_n)
        
        if not knn_recommendations:
            logger.warning(f"No KNN-based recommendations for user {user_id}. Trying association rules.")
            
            # Use association rules if KNN is unavailable
            association_recommendations = get_association_rule_recommendations(user_id, top_n)
            
            if not association_recommendations:
                logger.warning(f"No association rule recommendations for user {user_id}. Falling back to popular products.")
                
                # Fall back to popular products if association rules don't work
                popular_recommendations = get_popular_products(top_n)
                recommended_products = Product.objects.filter(id__in=popular_recommendations)
                
                return recommended_products
        
        recommended_products = Product.objects.filter(id__in=knn_recommendations)
        return recommended_products

    except Exception as e:
        logger.error(f"Error occurred while fetching combined recommendations for user {user_id}: {e}")
        return [] 

def get_popular_products(top_n=5):
    try:
        popular_products = Product.objects.annotate(purchase_count=Count('transaction')).order_by('-purchase_count')[:top_n]
        return [product.id for product in popular_products]

    except Exception as e:
        logger.error(f"Error occurred while fetching popular products: {e}")
        return [] 

def plot_popular_products(top_n=5):
    try:
        popular_products = Product.objects.annotate(purchase_count=Count('transaction')).order_by('-purchase_count')[:top_n]
        popular_product_names = [product.name for product in popular_products]
        purchase_counts = [product.purchase_count for product in popular_products]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=popular_product_names, y=purchase_counts)
        plt.title(f"Top {top_n} Popular Products")
        plt.xlabel("Product Name")
        plt.ylabel("Number of Purchases")
        plt.xticks(rotation=45, ha="right")

        # Save plot to a BytesIO object and encode as base64
        img_io = io.BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        plt.close()

        return img_base64

    except Exception as e:
        logger.error(f"Error occurred while plotting popular products: {e}")
        return None

def get_association_rule_recommendations(user_id, top_n=5):
    try:
        # Ensure you're querying the correct field for user_id
        transactions = Transaction.objects.filter(user__id=user_id).values('user_id', 'product_id')

        # Log the fetched transactions to verify what data is being retrieved
        logger.debug(f"Transactions fetched for user {user_id}: {transactions}")

        if not transactions:
            logger.warning(f"No transaction data found for user {user_id}.")
            return []

        transaction_df = pd.DataFrame(transactions)

        # Log the DataFrame for debugging
        logger.debug(f"Transaction DataFrame for user {user_id}:\n{transaction_df.head()}")

        # Check if 'user_id' is in the DataFrame columns
        if 'user_id' not in transaction_df.columns:
            logger.error(f"Column 'user_id' not found in the transaction data for user {user_id}.")
            return []

        # Continue with your logic...
    except Exception as e:
        logger.error(f"Error occurred while generating association rule recommendations for user {user_id}: {e}")
        return []


def evaluate_knn_recommendations(user_id, top_n=5):
 
    try:
        # Get KNN-based recommendations for the user
        recommended_product_ids = get_knn_recommendations(user_id, top_n)

        if not recommended_product_ids:
            return {'precision': 0, 'recall': 0, 'f1': 0}

        # Get the user's actual purchased products
        user_transactions = Transaction.objects.filter(user_id=user_id).values_list('product_id', flat=True)
        user_transactions_set = set(user_transactions)

        # Calculate Precision, Recall, and F1-Score
        true_positives = len(set(recommended_product_ids) & user_transactions_set)
        false_positives = len(set(recommended_product_ids) - user_transactions_set)
        false_negatives = len(user_transactions_set - set(recommended_product_ids))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    except Exception as e:
        logger.error(f"Error occurred while evaluating KNN recommendations for user {user_id}: {e}")
        return {'precision': 0, 'recall': 0, 'f1': 0}
def evaluate_association_rule_recommendations(user_id, top_n=5):

    try:
        # Get Association Rule-based recommendations for the user
        recommended_product_ids = get_association_rule_recommendations(user_id, top_n)

        if not recommended_product_ids:
            return {'precision': 0, 'recall': 0, 'f1': 0}

        # Get the user's actual purchased products
        user_transactions = Transaction.objects.filter(user_id=user_id).values_list('product_id', flat=True)
        user_transactions_set = set(user_transactions)

        # Calculate Precision, Recall, and F1-Score
        true_positives = len(set(recommended_product_ids) & user_transactions_set)
        false_positives = len(set(recommended_product_ids) - user_transactions_set)
        false_negatives = len(user_transactions_set - set(recommended_product_ids))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    except Exception as e:
        logger.error(f"Error occurred while evaluating Association Rule recommendations for user {user_id}: {e}")
        return {'precision': 0, 'recall': 0, 'f1': 0}

def evaluate_popular_products_recommendations(user_id, top_n=5):
    try:
        # Get the top N most popular products
        popular_product_ids = get_popular_products(top_n)

        if not popular_product_ids:
            return {'precision': 0, 'recall': 0, 'f1': 0}

        # Get the user's actual purchased products
        user_transactions = Transaction.objects.filter(user_id=user_id).values_list('product_id', flat=True)
        user_transactions_set = set(user_transactions)

        # Calculate Precision, Recall, and F1-Score
        true_positives = len(set(popular_product_ids) & user_transactions_set)
        false_positives = len(set(popular_product_ids) - user_transactions_set)
        false_negatives = len(user_transactions_set - set(popular_product_ids))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    except Exception as e:
        logger.error(f"Error occurred while evaluating Popular Products recommendations for user {user_id}: {e}")
        return {'precision': 0, 'recall': 0, 'f1': 0}
