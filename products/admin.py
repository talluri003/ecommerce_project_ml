from django.contrib import admin

from products.models import *



# Register models with their respective admin classes
admin.site.register(Product)
admin.site.register(Cart)
admin.site.register(CartItem)
admin.site.register(ProductLike)
admin.site.register(Transaction)
admin.site.register(Order)
admin.site.register(OrderTransaction)
admin.site.register(UserInteraction)
admin.site.register(RecommendationLog)
