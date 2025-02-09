import os
import logging
import time
import socket
import asyncio
import warnings
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, ContextTypes
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tensorflow import get_logger

# Suppress logging
get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WDM_LOG'] = str(logging.ERROR)
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration constants
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
GROUP_CHAT_ID = os.getenv('GROUP_CHAT_ID')
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}
TIMEOUT = 30
INFLUENCERS = [
    ("justsaurabh1103", "Saurabh Tiwari"),
    ("TheDustyBC", "DustyBC Crypto"),
    ("Trader_Jibon", "Trader_J"),
    ("cryptocevo", "Cevo"),
    ("WhalePanda", "WhalePanda"),
    ("loomdart", "Loomdart"),
    ("KoroushAK", "Koroush AK"),
    ("Tradermayne", "Mayne"),
    ("AltcoinGordon", "Gordon"),
    ("Trader_XO", "XO"),
    ("CryptoWizardd", "WIZZ"),
    ("MartiniGuyYT", "That Martini Guy ₿"),
]

class TwitterScraper:
    def __init__(self):
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        try:
            self.driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()),
                options=options
            )
            logger.info("WebDriver initialized")
        except Exception as e:
            logger.error(f"WebDriver init failed: {str(e)}")
            raise

    def get_recent_tweets(self, handle: str) -> list[dict]:
        logger.info(f"Scraping @{handle}")
        tweets = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        try:
            self.driver.get(f"https://nitter.net/{handle}")
            time.sleep(3)
            
            tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, ".timeline-item")[:5]
            logger.debug(f"Found {len(tweet_elements)} tweets")

            for tweet in tweet_elements:
                try:
                    content = tweet.find_element(By.CSS_SELECTOR, ".tweet-content").text
                    link = tweet.find_element(By.CSS_SELECTOR, "a.tweet-link").get_attribute("href")
                    tweet_id = link.split("/status/")[1].split("#")[0]
                    
                    date_str = tweet.find_element(By.CSS_SELECTOR, ".tweet-date a").get_attribute("title")
                    tweet_time = datetime.strptime(
                        date_str.replace(" UTC", ""), 
                        "%b %d, %Y · %I:%M %p"
                    ).replace(tzinfo=timezone.utc)
                    
                    if tweet_time > cutoff_time:
                        tweets.append({
                            'text': content[:500],
                            'id': tweet_id,
                            'handle': handle,
                            'time': tweet_time
                        })
                        logger.info(f"New tweet @{tweet_time.isoformat()}")
                        
                except (NoSuchElementException, ValueError) as e:
                    logger.debug(f"Skipping tweet: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Scraping error: {str(e)}")
            
        logger.info(f"Found {len(tweets)} valid tweets")
        return tweets

    def close(self):
        try:
            self.driver.quit()
            logger.debug("WebDriver closed")
        except Exception as e:
            logger.error(f"Close error: {str(e)}")

async def check_network_connection():
    try:
        reader, writer = await asyncio.open_connection("api-inference.huggingface.co", 443)
        writer.close()
        await writer.wait_closed()
        return True
    except (socket.gaierror, OSError) as e:
        logger.error(f"Network error: {str(e)}")
        return False

def create_prompt(text: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>"
        "Analyze if this tweet explicitly recommends buying cryptocurrency. "
        "Respond ONLY with 'YES' or 'NO'.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>"
        f"{text[:500]}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
)
async def is_buy_signal(text: str) -> bool:
    if not await check_network_connection():
        return False

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                HF_API_URL,
                headers=HEADERS,
                json={
                    "inputs": create_prompt(text),
                    "parameters": {
                        "max_new_tokens": 10,
                        "temperature": 0.1,
                        "return_full_text": False
                    }
                },
                timeout=TIMEOUT
            )
            
        if response.status_code == 200:
            result = response.json()
            if not isinstance(result, list) or len(result) == 0:
                logger.error("Invalid API response")
                return False
                
            answer = result[0].get('generated_text', '').strip().upper()
            logger.info(f"API response: {answer}")
            return answer == 'YES'
            
        logger.error(f"API error {response.status_code}: {response.text[:200]}")
        if response.status_code == 401:
            logger.error("Invalid API token")
        elif response.status_code == 404:
            logger.error("Model not found")
            
        return False
        
    except httpx.RequestError as e:
        logger.error(f"Connection error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return False

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Telegram error: {context.error}", exc_info=context.error)

async def check_influencers(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Starting monitoring cycle")
    scraper = TwitterScraper()
    
    try:
        for handle, name in INFLUENCERS:
            logger.info(f"Checking {name}")
            tweets = scraper.get_recent_tweets(handle)
            
            if not tweets:
                logger.debug("No recent tweets")
                continue
                
            logger.info(f"Analyzing {len(tweets)} tweets")
            for tweet in tweets:
                logger.info(f"Tweet @{tweet['time'].isoformat()}")
                try:
                    is_buy = await is_buy_signal(tweet['text'])
                    logger.info(f"Buy signal: {is_buy}")
                    if is_buy:
                        message = (
                            f"🚨 BUY ALERT from {name}\n"
                            f"📅 {tweet['time'].strftime('%Y-%m-%d %H:%M UTC')}\n"
                            f"📝 {tweet['text'][:200]}...\n"
                            f"🔗 https://twitter.com/{handle}/status/{tweet['id']}"
                        )
                        await context.bot.send_message(GROUP_CHAT_ID, message)
                        logger.info("Alert sent")
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Cycle error: {str(e)}")
    finally:
        scraper.close()
        logger.info("Monitoring complete")

def main():
    required_vars = ['TELEGRAM_BOT_TOKEN', 'HF_API_TOKEN', 'GROUP_CHAT_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing: {', '.join(missing_vars)}")
        return

    try:
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_error_handler(error_handler)
        app.job_queue.run_repeating(check_influencers, interval=3600, first=10)
        logger.info("Bot started")
        app.run_polling()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")

if __name__ == '__main__':
    main()