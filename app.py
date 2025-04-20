from flask import Flask, render_template
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def show_predictions():
    # Your prediction data (now dynamic)
    predictions = [
        {"date": (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
         "price": round(price, 2)}
        for i, price in enumerate([
            4148.147289, 3997.882335, 3937.439923,
            3868.828430, 3826.884528, 3792.975894, 3796.002389
        ])
    ]
    
    # Model execution times
    exec_times = [
        "1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 506ms/step (initial)",
        "Subsequent predictions: 36ms-84ms/step"
    ]
    
    return render_template(
        'index.html',
        commodity="Soyabean",
        predictions=predictions,
        exec_times=exec_times
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
