import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from mlxtend.frequent_patterns import apriori, association_rules
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from django.db.models import Count
from products.models import Cart, Transaction, Product
import io
import base64
import matplotlib
matplotlib.use('Agg')  # To avoid issues with TkAgg (default matplotlib backend)

# Set up logging
logger = logging.getLogger(__name__)
def create_user_item_matrix():
    try:
        transactions = Transaction.objects.all().values('user_id', 'product_id')
        transaction_df = pd.DataFrame(transactions)

        if transaction_df.empty:
            logger.warning("No transaction data found.")
            return pd.DataFrame(), None
        
        transaction_df['interaction'] = 1  # Purchases get interaction score of 1
        
        cart_items = Cart.objects.all().values('user_id', 'product_id')
        cart_df = pd.DataFrame(cart_items)

        if cart_df.empty:
            logger.warning("No cart data found.")
            return pd.DataFrame(), None
        
        cart_df['interaction'] = 0.5  # Cart items get lower interaction score of 0.5
        
        df = pd.concat([transaction_df[['user_id', 'product_id', 'interaction']], 
                        cart_df[['user_id', 'product_id', 'interaction']]], ignore_index=True)
        
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

from sklearn.neighbors import NearestNeighbors
import pandas as pd

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

        user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)

        knn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine')
        
        # Fit KNN on the transposed user-item matrix
        knn.fit(user_item_matrix.values.T)  # Each product should be a row, each user a column

        distances, indices = knn.kneighbors(user_vector)

        recommended_product_ids = user_item_matrix.columns[indices.flatten()[1:]].tolist()

        logger.info(f"KNN recommendations for user {user_id}: {recommended_product_ids}")

        return recommended_product_ids

    except Exception as e:
        logger.error(f"Error occurred while generating KNN recommendations: {e}")
        return []


def get_combined_recommendations(user_id, top_n=5):
    try:
        # Get KNN-based recommendations
        knn_recommendations = get_knn_recommendations(user_id, top_n)

        if not knn_recommendations:
            logger.warning(f"No KNN-based recommendations found for user {user_id}. Falling back to association rules.")
            
            # If KNN fails, use Association Rules to generate recommendations
            association_recommendations = get_association_rule_recommendations(user_id, top_n)

            if not association_recommendations:
                logger.warning(f"No recommendations found using association rules for user {user_id}. Falling back to popular products.")
                
                # If association rules fail, fall back to popular products
                popular_recommendations = get_popular_products(top_n)
                plot_popular_products(top_n)  # Visualize popular products
                recommended_products = Product.objects.filter(id__in=popular_recommendations)
                return recommended_products

            # If association rule recommendations are found, use them
            return association_recommendations

        # If KNN recommendations are found, use them
        recommended_products = Product.objects.filter(id__in=knn_recommendations)
        return recommended_products

    except Exception as e:
        logger.error(f"Error occurred while fetching combined recommendations for user {user_id}: {e}")
        return []  # Return an empty list if any error occurs

def get_popular_products(top_n=5):
    try:
        popular_products = Product.objects.annotate(purchase_count=Count('transaction')).order_by('-purchase_count')[:top_n]
        return [product.id for product in popular_products]

    except Exception as e:
        logger.error(f"Error occurred while fetching popular products: {e}")
        return []  # Return an empty list if any error occurs

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
        #  Fetch transaction data (user_id, product_id)
        transactions = Transaction.objects.filter(user_id=user_id).values('product_id')
        transaction_df = pd.DataFrame(transactions)

        if transaction_df.empty:
            logger.warning(f"No transaction data found for user {user_id}.")
            return []

        #  Create a binary user-item matrix
        # Create a binary matrix where 1 indicates a product was purchased, 0 otherwise
        user_item_matrix = transaction_df.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
        
        # Convert to binary format (0 or 1) to comply with the Apriori algorithm
        user_item_matrix[user_item_matrix > 0] = 1  # Set all values greater than 0 to 1 (purchase occurred)

        # Ensure binary values (0 or 1)
        user_item_matrix = user_item_matrix.astype(int)

        # Apply the Apriori algorithm to get frequent itemsets
        frequent_itemsets = apriori(user_item_matrix, min_support=0.1, use_colnames=True)
        
        # Generate association rules from the frequent itemsets
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        if rules.empty:
            logger.warning(f"No association rules found for user {user_id}.")
            return []

        # Sort the rules by confidence (or lift) and recommend top N products
        rules = rules.sort_values(by='confidence', ascending=False)
        recommended_product_ids = rules['consequents'].head(top_n).apply(lambda x: list(x)[0]).tolist()

        logger.info(f"Association rule recommendations for user {user_id}: {recommended_product_ids}")

        return recommended_product_ids

    except Exception as e:
        logger.error(f"Error occurred while generating association rule recommendations for user {user_id}: {e}")
        return []  # Return an empty list if any error occurs


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
