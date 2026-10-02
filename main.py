import os
import re
import uuid
import sqlite3
import hashlib
import secrets
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File
)

from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    JSONResponse
)

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware


# =========================================================
# 경로
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
PROFILE_DIR = STATIC_DIR / "profiles"

CSV_PATH = STATIC_DIR / "예금목록.csv"
DB_PATH = BASE_DIR / "users.db"


STATIC_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# ENV
# =========================================================

load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="하나 금융상품 추천",
    version="1.0.0"
)


app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv(
        "SESSION_SECRET",
        "hana-finance-secret-key"
    ),
    max_age=60 * 60 * 24 * 7
)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)


# =========================================================
# SQLite
# =========================================================

def get_db():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            investment_type TEXT,
            job TEXT,
            income INTEGER,
            profile_image TEXT,
            created_at TEXT
        )
        """
    )

    conn.commit()

    conn.close()


init_db()


# =========================================================
# 비밀번호
# =========================================================

def hash_password(password):

    salt = secrets.token_hex(16)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    )

    return f"{salt}${hashed.hex()}"


def verify_password(password, saved_password):

    try:

        salt, saved_hash = saved_password.split("$")

        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000
        )

        return secrets.compare_digest(
            hashed.hex(),
            saved_hash
        )

    except Exception:

        return False


# =========================================================
# 현재 로그인 사용자
# =========================================================

def get_current_user(request):

    user_id = request.session.get("user_id")

    if not user_id:
        return None

    conn = get_db()

    user = conn.execute(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    conn.close()

    if not user:
        return None

    return dict(user)


# =========================================================
# CSV
# =========================================================

def load_products():

    if not CSV_PATH.exists():

        print(
            f"[오류] CSV가 없습니다: {CSV_PATH}"
        )

        return pd.DataFrame()


    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr"
    ]


    for encoding in encodings:

        try:

            df = pd.read_csv(
                CSV_PATH,
                encoding=encoding
            )

            df = df.fillna("")

            return df

        except UnicodeDecodeError:

            continue

        except Exception as e:

            print(
                "CSV 읽기 오류:",
                e
            )

            return pd.DataFrame()


    return pd.DataFrame()


# =========================================================
# 안전하게 값 추출
# =========================================================

def safe_text(row, column, default=""):

    try:

        value = row.get(
            column,
            default
        )

        if pd.isna(value):
            return default

        return str(value).strip()

    except Exception:

        return default


def to_float(value):

    if value is None:
        return 0.0

    text = str(value)

    match = re.search(
        r"\d+(?:\.\d+)?",
        text
    )

    if not match:
        return 0.0

    try:

        return float(
            match.group()
        )

    except Exception:

        return 0.0


def get_rate(row, numeric_column, text_column):

    numeric_value = safe_text(
        row,
        numeric_column
    )

    if numeric_value:

        return to_float(
            numeric_value
        )

    return to_float(
        safe_text(
            row,
            text_column
        )
    )


# =========================================================
# 맞춤형 상품 추천
# =========================================================

def recommend_products(user, count=3):

    df = load_products()

    if df.empty:
        return []


    age = int(
        user.get("age") or 0
    )

    investment_type = str(
        user.get(
            "investment_type"
        ) or ""
    )

    job = str(
        user.get("job") or ""
    )

    income = int(
        user.get("income") or 0
    )


    products = []


    for _, row in df.iterrows():

        bank = safe_text(
            row,
            "금융사"
        )

        product_name = safe_text(
            row,
            "상품명"
        )

        max_rate = get_rate(
            row,
            "최고금리_숫자",
            "최고금리"
        )

        basic_rate = get_rate(
            row,
            "기본금리_숫자",
            "기본금리"
        )

        max_rate_text = safe_text(
            row,
            "최고금리"
        )

        basic_rate_text = safe_text(
            row,
            "기본금리"
        )

        period = safe_text(
            row,
            "가입기간"
        )

        amount = safe_text(
            row,
            "가입금액"
        )

        method = safe_text(
            row,
            "가입방법"
        )

        target = safe_text(
            row,
            "가입대상"
        )

        product_type = safe_text(
            row,
            "상품유형"
        )

        detail = safe_text(
            row,
            "상세정보전체"
        )

        url = safe_text(
            row,
            "상세URL"
        )


        full_text = (
            f"{bank} "
            f"{product_name} "
            f"{period} "
            f"{amount} "
            f"{method} "
            f"{target} "
            f"{product_type} "
            f"{detail}"
        ).lower()


        score = 0

        reasons = []


        # =================================================
        # 금리
        # =================================================

        score += max_rate * 15

        score += basic_rate * 5


        if max_rate >= 3.5:

            reasons.append(
                f"최고금리 {max_rate:.2f}% 수준의 상품"
            )


        # =================================================
        # 투자성향
        # =================================================

        if investment_type == "안정형":

            score += basic_rate * 6

            if (
                "예금" in full_text
                or "정기" in full_text
            ):
                score += 10

            reasons.append(
                "안정성을 중시하는 투자성향을 고려"
            )


        elif investment_type == "안정추구형":

            score += basic_rate * 4

            score += max_rate * 2

            reasons.append(
                "안정성과 금리를 함께 고려"
            )


        elif investment_type == "위험중립형":

            score += max_rate * 5

            reasons.append(
                "금리 경쟁력을 중심으로 선정"
            )


        elif investment_type == "적극투자형":

            score += max_rate * 7

            reasons.append(
                "상대적으로 높은 금리를 우선 고려"
            )


        elif investment_type == "공격투자형":

            score += max_rate * 9

            reasons.append(
                "예금상품 중 높은 금리를 우선 고려"
            )


        # =================================================
        # 나이
        # =================================================

        if age <= 29:

            keywords = [
                "청년",
                "첫거래",
                "첫 거래",
                "학생",
                "모바일"
            ]

            for keyword in keywords:

                if keyword in full_text:

                    score += 10

                    reasons.append(
                        "청년·사회초년생 관련 조건을 고려"
                    )

                    break


        elif 30 <= age <= 49:

            keywords = [
                "직장",
                "급여",
                "정기"
            ]

            for keyword in keywords:

                if keyword in full_text:

                    score += 5

                    break


        elif age >= 50:

            keywords = [
                "연금",
                "노후",
                "시니어",
                "은퇴"
            ]

            for keyword in keywords:

                if keyword in full_text:

                    score += 10

                    reasons.append(
                        "노후 자금 관리와 관련된 조건을 고려"
                    )

                    break


        # =================================================
        # 직업
        # =================================================

        job_keywords = {

            "직장인": [
                "직장",
                "급여",
                "월급"
            ],

            "공무원": [
                "공무원",
                "급여"
            ],

            "군인": [
                "군인",
                "장병",
                "군"
            ],

            "학생": [
                "학생",
                "청년",
                "첫거래"
            ],

            "자영업자": [
                "사업자",
                "소상공인",
                "사업"
            ],

            "프리랜서": [
                "비대면",
                "모바일"
            ]
        }


        for keyword in job_keywords.get(
            job,
            []
        ):

            if keyword in full_text:

                score += 10

                reasons.append(
                    f"{job} 고객과 관련된 상품 조건"
                )

                break


        # =================================================
        # 소득
        # =================================================

        if income < 30000000:

            if (
                "소액" in full_text
                or "청년" in full_text
                or "첫거래" in full_text
            ):

                score += 5


        elif income < 70000000:

            score += basic_rate * 1.5


        else:

            score += max_rate * 2


        # 비대면
        if (
            "비대면" in full_text
            or "인터넷" in full_text
            or "스마트폰" in full_text
            or "모바일" in full_text
        ):

            score += 2


        # 이유 중복 제거
        unique_reasons = []

        for reason in reasons:

            if reason not in unique_reasons:

                unique_reasons.append(
                    reason
                )


        if not unique_reasons:

            unique_reasons.append(
                "회원정보와 상품의 금리 및 가입조건을 종합적으로 고려"
            )


        if not max_rate_text:

            max_rate_text = (
                f"{max_rate:.2f}%"
                if max_rate
                else "-"
            )


        if not basic_rate_text:

            basic_rate_text = (
                f"{basic_rate:.2f}%"
                if basic_rate
                else "-"
            )


        products.append(
            {
                "bank": bank or "금융사",
                "name": product_name or "상품명",
                "max_rate": max_rate_text,
                "basic_rate": basic_rate_text,
                "period": period,
                "amount": amount,
                "method": method,
                "target": target,
                "product_type": product_type,
                "url": url,
                "score": score,
                "reason": unique_reasons[0]
            }
        )


    # 점수 순 정렬
    products.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    # 중복 상품 제거
    result = []

    seen = set()


    for product in products:

        key = (
            product["bank"],
            product["name"]
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(product)

        if len(result) >= count:
            break


    return result


# =========================================================
# 챗봇에 넘길 상품 검색
# =========================================================

def search_products_for_chat(
    message,
    user,
    limit=12
):

    df = load_products()

    if df.empty:

        return (
            "현재 금융상품 CSV 데이터를 "
            "불러오지 못했습니다."
        )


    keywords = re.findall(
        r"[가-힣A-Za-z0-9.%]+",
        message.lower()
    )


    keywords = [
        keyword
        for keyword in keywords
        if len(keyword) >= 2
    ]


    candidates = []


    for _, row in df.iterrows():

        bank = safe_text(
            row,
            "금융사"
        )

        name = safe_text(
            row,
            "상품명"
        )

        basic_rate = safe_text(
            row,
            "기본금리"
        )

        max_rate = safe_text(
            row,
            "최고금리"
        )

        period = safe_text(
            row,
            "가입기간"
        )

        amount = safe_text(
            row,
            "가입금액"
        )

        method = safe_text(
            row,
            "가입방법"
        )

        target = safe_text(
            row,
            "가입대상"
        )

        product_type = safe_text(
            row,
            "상품유형"
        )

        detail = safe_text(
            row,
            "상세정보전체"
        )

        url = safe_text(
            row,
            "상세URL"
        )


        searchable = (
            f"{bank} "
            f"{name} "
            f"{basic_rate} "
            f"{max_rate} "
            f"{period} "
            f"{amount} "
            f"{method} "
            f"{target} "
            f"{product_type} "
            f"{detail}"
        ).lower()


        score = 0


        for keyword in keywords:

            if keyword in searchable:

                score += 8


        max_rate_num = get_rate(
            row,
            "최고금리_숫자",
            "최고금리"
        )


        # "금리", "높은", "고금리" 질문이면 고금리 우선
        if any(
            keyword in message
            for keyword in [
                "고금리",
                "금리",
                "높은",
                "최고금리"
            ]
        ):

            score += (
                max_rate_num * 10
            )

        else:

            score += max_rate_num


        candidates.append(
            (
                score,
                max_rate_num,
                {
                    "bank": bank,
                    "name": name,
                    "basic_rate": basic_rate,
                    "max_rate": max_rate,
                    "period": period,
                    "amount": amount,
                    "method": method,
                    "target": target,
                    "product_type": product_type,
                    "detail": detail,
                    "url": url
                }
            )
        )


    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        ),
        reverse=True
    )


    selected = candidates[:limit]


    result = []


    for index, (
        score,
        max_rate_num,
        product
    ) in enumerate(
        selected,
        start=1
    ):

        result.append(
            f"""
[상품 {index}]
금융사: {product['bank']}
상품명: {product['name']}
기본금리: {product['basic_rate']}
최고금리: {product['max_rate']}
가입기간: {product['period']}
가입금액: {product['amount']}
가입방법: {product['method']}
가입대상: {product['target']}
상품유형: {product['product_type']}
상세정보: {product['detail']}
상세URL: {product['url']}
""".strip()
        )


    return "\n\n".join(result)


# =========================================================
# 루트
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def root(request: Request):

    user = get_current_user(
        request
    )

    if user:

        return RedirectResponse(
            url="/main",
            status_code=302
        )

    return RedirectResponse(
        url="/login",
        status_code=302
    )


# =========================================================
# 로그인
# =========================================================

@app.get(
    "/login",
    response_class=HTMLResponse
)
async def login_page(
    request: Request
):

    if get_current_user(request):

        return RedirectResponse(
            url="/main",
            status_code=302
        )

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={
            "error": None
        }
    )


@app.post(
    "/login",
    response_class=HTMLResponse
)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):

    conn = get_db()

    user = conn.execute(
        """
        SELECT *
        FROM users
        WHERE username = ?
        """,
        (username.strip(),)
    ).fetchone()

    conn.close()


    if (
        not user
        or not verify_password(
            password,
            user["password"]
        )
    ):

        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error":
                "아이디 또는 비밀번호가 올바르지 않습니다."
            },
            status_code=400
        )


    request.session[
        "user_id"
    ] = user["id"]


    return RedirectResponse(
        url="/main",
        status_code=303
    )


# =========================================================
# 회원가입
# =========================================================

@app.get(
    "/signup",
    response_class=HTMLResponse
)
async def signup_page(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={
            "error": None
        }
    )


@app.post(
    "/signup",
    response_class=HTMLResponse
)
async def signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    gender: str = Form(...),
    age: int = Form(...),
    investment_type: str = Form(...),
    job: str = Form(...),
    income: int = Form(...),
    profile_image: UploadFile = File(None)
):

    username = username.strip()

    email = email.strip()


    if len(username) < 4:

        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={
                "error":
                "아이디는 4자 이상 입력해주세요."
            },
            status_code=400
        )


    if len(password) < 4:

        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={
                "error":
                "비밀번호는 4자 이상 입력해주세요."
            },
            status_code=400
        )


    profile_path = (
        "/static/logo.png"
    )


    # 프로필 이미지 저장
    if (
        profile_image
        and profile_image.filename
    ):

        extension = Path(
            profile_image.filename
        ).suffix.lower()


        allowed_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp"
        }


        if extension in allowed_extensions:

            filename = (
                f"{uuid.uuid4().hex}"
                f"{extension}"
            )

            save_path = (
                PROFILE_DIR / filename
            )

            contents = (
                await profile_image.read()
            )

            with open(
                save_path,
                "wb"
            ) as f:

                f.write(contents)


            profile_path = (
                f"/static/profiles/"
                f"{filename}"
            )


    conn = get_db()


    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users (
                username,
                password,
                name,
                email,
                gender,
                age,
                investment_type,
                job,
                income,
                profile_image,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                hash_password(password),
                name,
                email,
                gender,
                age,
                investment_type,
                job,
                income,
                profile_path,
                datetime.now().isoformat()
            )
        )

        conn.commit()

        user_id = (
            cursor.lastrowid
        )


    except sqlite3.IntegrityError:

        conn.close()

        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context={
                "error":
                "이미 사용 중인 아이디입니다."
            },
            status_code=400
        )


    finally:

        try:
            conn.close()
        except Exception:
            pass


    request.session[
        "user_id"
    ] = user_id


    return RedirectResponse(
        url="/main",
        status_code=303
    )


# =========================================================
# 메인
# =========================================================

@app.get(
    "/main",
    response_class=HTMLResponse
)
async def main_page(
    request: Request
):

    user = get_current_user(
        request
    )


    if not user:

        return RedirectResponse(
            url="/login",
            status_code=302
        )


    products = recommend_products(
        user,
        count=3
    )


    return templates.TemplateResponse(
        request=request,
        name="main.html",
        context={
            "user": user,
            "products": products
        }
    )


# =========================================================
# 챗봇 페이지
# =========================================================

@app.get(
    "/chatbot",
    response_class=HTMLResponse
)
async def chatbot_page(
    request: Request
):

    user = get_current_user(
        request
    )


    if not user:

        return RedirectResponse(
            url="/login",
            status_code=302
        )


    return templates.TemplateResponse(
        request=request,
        name="chatbot.html",
        context={
            "user": user
        }
    )


# =========================================================
# 챗봇 API
# =========================================================

@app.post("/api/chat")
async def chat_api(
    request: Request
):

    user = get_current_user(
        request
    )


    if not user:

        return JSONResponse(
            {
                "error":
                "로그인이 필요합니다."
            },
            status_code=401
        )


    if client is None:

        return JSONResponse(
            {
                "error":
                ".env 파일에 OPENAI_API_KEY를 설정해주세요."
            },
            status_code=500
        )


    try:

        body = await request.json()

        message = str(
            body.get(
                "message",
                ""
            )
        ).strip()


        history = body.get(
            "history",
            []
        )


        if not message:

            return JSONResponse(
                {
                    "error":
                    "메시지를 입력해주세요."
                },
                status_code=400
            )


        product_context = (
            search_products_for_chat(
                message,
                user,
                limit=12
            )
        )


        income_value = int(
            user.get("income") or 0
        )


        user_info = f"""
이름: {user.get('name')}
나이: {user.get('age')}세
성별: {user.get('gender')}
투자성향: {user.get('investment_type')}
직업: {user.get('job')}
연소득: {income_value:,}원
""".strip()


        system_prompt = f"""
너는 금융 예금상품 상담 AI '하나봇'이다.

사용자의 회원정보와 실제 CSV 금융상품 데이터를 이용하여
예금상품을 설명하고 비교한다.

[현재 사용자 정보]

{user_info}


[현재 질문과 관련된 금융상품 데이터]

{product_context}


반드시 다음 규칙을 따른다.

1. 상품정보는 위 CSV 데이터에 존재하는 내용을 우선 사용한다.

2. CSV에 없는 금리, 가입조건, 우대조건을 임의로 만들어내지 않는다.

3. 최고금리는 우대조건을 충족해야 적용될 수 있으므로,
   필요한 경우 실제 상품설명서를 확인하도록 안내한다.

4. 사용자가 상품을 추천해달라고 하면
   사용자의 나이, 투자성향, 직업, 소득을 고려하여 설명한다.

5. 사용자가 여러 상품을 추천하거나 비교해달라고 하면
   반드시 Markdown 표를 사용한다.

6. Markdown 표는 반드시 아래 형식과 동일한 문법을 사용한다.

| 금융사 | 상품명 | 기본금리 | 최고금리 | 가입기간 | 가입금액 |
|---|---|---:|---:|---|---|
| 은행명 | 상품명 | 3.00% | 3.50% | 12개월 | 100만원 이상 |

7. 표 안에는 Markdown 줄바꿈 태그나 HTML을 사용하지 않는다.

8. 금융상품이 여러 개라면 상품마다 한 행씩 작성한다.

9. 추천 상품은 가능하면 3~5개 정도만 보여준다.

10. 표 다음에는 추천 이유를 자연스러운 문장으로 설명한다.

11. 중요한 상품명이나 금리는
    **굵은 글씨** 형식으로 강조할 수 있다.

12. 답변은 한국어로 작성한다.

13. 답변을 지나치게 길게 작성하지 않는다.

14. 특정 금융상품 가입을 강요하지 않는다.

15. 공격투자형 사용자라고 해서
    예금 자체가 공격적인 투자상품이라고 표현하지 않는다.
    예금상품 안에서 상대적으로 높은 금리를 선호할 가능성을
    고려하는 수준으로만 설명한다.

16. 사용자가 특정 상품을 물으면
    해당 상품의 금융사, 기본금리, 최고금리,
    가입기간, 가입금액, 가입방법 등을 설명한다.

17. 최종 가입 전 해당 금융사의 공식 홈페이지나
    상품설명서에서 최신 조건을 확인하도록 안내한다.

18. 금융상품과 관계없는 질문에는
    예금 및 금융상품 상담을 도와드릴 수 있다고 짧게 안내한다.
""".strip()


        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]


        if isinstance(
            history,
            list
        ):

            for item in history[-12:]:

                if not isinstance(
                    item,
                    dict
                ):
                    continue


                role = item.get(
                    "role"
                )

                content = str(
                    item.get(
                        "content",
                        ""
                    )
                )[:3000]


                if (
                    role
                    in [
                        "user",
                        "assistant"
                    ]
                    and content
                ):

                    messages.append(
                        {
                            "role": role,
                            "content": content
                        }
                    )


        messages.append(
            {
                "role": "user",
                "content": message
            }
        )


        response = (
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )
        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        return {
            "answer": answer
        }


    except Exception as e:

        print(
            "챗봇 오류:",
            str(e)
        )


        return JSONResponse(
            {
                "error":
                f"챗봇 처리 중 오류가 발생했습니다: {str(e)}"
            },
            status_code=500
        )


# =========================================================
# 로그아웃
# =========================================================

@app.get("/logout")
async def logout(
    request: Request
):

    request.session.clear()

    return RedirectResponse(
        url="/login",
        status_code=302
    )


# =========================================================
# 실행
# =========================================================

if __name__ == "__main__":

    import uvicorn

    print()
    print(
        "=" * 60
    )
    print(
        "하나 금융상품 추천 서비스"
    )
    print(
        "http://localhost:1122"
    )
    print(
        "=" * 60
    )
    print()

    uvicorn.run(
        "main:app",
        host="localhost",
        port=1122,
        reload=True
    )