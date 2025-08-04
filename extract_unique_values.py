#!/usr/bin/env python3
"""
Extract unique values from flights.json for dynamic filter generation.
"""

import json
from collections import defaultdict

def extract_unique_values():
    """Extract unique values from flights.json for each filterable field."""
    try:
        with open('data/flights.json', 'r', encoding='utf-8') as file:
            flights = json.load(file)
        
        unique_values = defaultdict(set)
        
        for flight in flights:
            # Extract unique values for each field
            unique_values['airline'].add(flight.get('airline', ''))
            unique_values['alliance'].add(flight.get('alliance', ''))
            unique_values['from_country'].add(flight.get('from_country', ''))
            unique_values['to_country'].add(flight.get('to_country', ''))
            unique_values['travel_class'].add(flight.get('travel_class', ''))
            unique_values['refundable'].add(flight.get('refundable', False))
            unique_values['baggage_included'].add(flight.get('baggage_included', False))
            unique_values['wifi_available'].add(flight.get('wifi_available', False))
            unique_values['meal_service'].add(flight.get('meal_service', ''))
            unique_values['aircraft_type'].add(flight.get('aircraft_type', ''))
        
        # Convert sets to sorted lists
        filter_options = {}
        for key, values in unique_values.items():
            filter_options[key] = sorted(list(values))
        
        # Add price ranges
        prices = [flight.get('price_usd', 0) for flight in flights]
        filter_options['price_ranges'] = {
            'min': min(prices),
            'max': max(prices),
            'suggested_ranges': [
                {'min': 0, 'max': 500, 'label': 'Budget (0-500 USD)'},
                {'min': 500, 'max': 1000, 'label': 'Economy (500-1000 USD)'},
                {'min': 1000, 'max': 2000, 'label': 'Mid-range (1000-2000 USD)'},
                {'min': 2000, 'max': 5000, 'label': 'Premium (2000-5000 USD)'},
                {'min': 5000, 'max': max(prices), 'label': 'Luxury (5000+ USD)'}
            ]
        }
        
        return filter_options
        
    except Exception as e:
        print(f"Error extracting unique values: {e}")
        return {}

if __name__ == "__main__":
    filter_options = extract_unique_values()
    print("Available filter options:")
    print(json.dumps(filter_options, indent=2, default=str)) 