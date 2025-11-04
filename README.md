# AI Trading Bot

*A humble template for AI-powered cryptocurrency trading bot using simple LLM-based market analysis.*

---

## 🛠️ Setup

```bash
git clone https://github.com/RobotTraders/AITradingBot.git
cd AITradingBot
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```

Configure your API keys in `.env`:
```bash
cp .env.template .env
nano .env
```

Add your keys:
```env
LLM_API_KEY=your_deepseek_api_key_here
EXCHANGE_API_KEY=your_bitunix_api_key_here
EXCHANGE_API_SECRET=your_bitunix_api_secret_here
```

---

## ⏰ Automated Trading (Cron Job)

To run the bot daily at midnight UTC:

```bash
crontab -e
```

Add this line:
```
# Cron format: minute - hour - day of month - month - day of week
0 0 * * * cd /path/to/AITradingBot && bash batch_runner.sh >> cron.log 2>&1
```

---

## ✅ Requirements

Python 3.12+
See `requirements.txt` for specific Python packages

---

## 📃 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## ⚠️ Disclaimer

All this material and related videos are for educational and entertainment purposes only. It is not financial advice nor an endorsement of any provider, product or service. The user bears sole responsibility for any actions taken based on this information, and Robot Traders and its affiliates will not be held liable for any losses or damages resulting from its use.



