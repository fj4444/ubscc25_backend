import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any

class TradingBot:
    def __init__(self, data):
        # Keywords that typically indicate positive or negative sentiment
        self.positive_keywords = [
            'adopt', 'approve', 'bull', 'buy', 'growth', 'institutional', 'investment',
            'long', 'partnership', 'positive', 'rally', 'recovery', 'support', 'upgrade',
            'win', 'success', 'breakthrough', 'innovation', 'adoption', 'integration'
        ]
        
        self.negative_keywords = [
            'ban', 'bear', 'crash', 'decline', 'drop', 'fraud', 'hack', 'investigation',
            'lawsuit', 'negative', 'regulation', 'reject', 'sell', 'short', 'warning',
            'loss', 'failure', 'security breach', 'scam', 'volatility'
        ]
        
        # Specific influential sources
        self.high_impact_sources = ['Twitter', 'Bloomberg', 'Reuters', 'CNBC', 'Coindesk']
        self.news_event = data
    
    def analyze_sentiment(self, title: str, source: str) -> float:
        """Analyze sentiment score based on title and source"""
        title_lower = title.lower()
        
        # Calculate positive and negative scores
        positive_score = sum(1 for word in self.positive_keywords if word in title_lower)
        negative_score = sum(1 for word in self.negative_keywords if word in title_lower)
        
        # Source credibility multiplier
        source_multiplier = 1.5 if source in self.high_impact_sources else 1.0
        
        # Calculate final sentiment score
        sentiment_score = (positive_score - negative_score) * source_multiplier
        
        # Check for strong indicators
        if any(word in title_lower for word in ['trump', 'president', 'executive order']):
            sentiment_score += 2.0
        
        return sentiment_score
    
    def analyze_price_action(self, previous_candles: List[Dict], observation_candles: List[Dict]) -> Dict[str, float]:
        """Analyze price action and technical indicators"""
        if not previous_candles or not observation_candles:
            return {'trend': 0, 'volatility': 0, 'momentum': 0}
        
        # Extract prices
        prev_closes = [candle['close'] for candle in previous_candles]
        obs_closes = [candle['close'] for candle in observation_candles]
        all_closes = prev_closes + obs_closes
        
        # Calculate simple moving average
        sma_short = np.mean(prev_closes[-3:]) if len(prev_closes) >= 3 else prev_closes[-1]
        sma_long = np.mean(prev_closes) if prev_closes else 0
        
        # Price trend (1 for uptrend, -1 for downtrend)
        trend = obs_closes[-1] / (prev_closes[0] + 1e-12) - 1
        
        # Volatility (standard deviation of returns)
        returns = np.diff(all_closes) / all_closes[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Momentum (recent price change)
        momentum = (obs_closes[-1] - prev_closes[0]) / prev_closes[0] if prev_closes[0] != 0 else 0
        
        return {
            'trend': trend,
            'volatility': volatility,
            'momentum': momentum,
            'sma_short': sma_short,
            'sma_long': sma_long
        }
    
    def metrics(self, news_event: Dict):
        """Make trading decision based on news event analysis"""
        sentiment_score = self.analyze_sentiment(news_event['title'], news_event['source'])
        price_analysis = self.analyze_price_action(
            news_event['previous_candles'], 
            news_event['observation_candles']
        )
        
        return sentiment_score, price_analysis['trend'], price_analysis['momentum'], price_analysis['volatility']
    
def trading_bot_score(data):
    bot = TradingBot(data)
    score, trend, mm, vol = bot.metrics(data)
    basic = np.sqrt(np.abs(score)) * trend * 1e3
    if abs(mm) < 1e-2:
        basic = -basic
    basic *= vol * 1e3
    return basic

def get_top50_abs_indices(lst):
    if len(lst) <= 50:
        return list(range(len(lst)))
    abs_with_indices = [(abs(x), i) for i, x in enumerate(lst)]
    abs_with_indices.sort(key=lambda x: x[0], reverse=True)
    top50_indices = [index for _, index in abs_with_indices[:50]]
    return top50_indices

def final_trading(json_data):
    scores = list()
    ids = list()
    for data in json_data:
        ids.append(data["id"])
        scores.append(trading_bot_score(data))
    indices = get_top50_abs_indices(scores)
    ans = list()
    for i in indices:
        d = dict()
        d["id"] = ids[i]
        d["decision"] = "SHORT"
        if scores[i] > 0:
            d["decision"] = "LONG"
        ans.append(d)
    return ans