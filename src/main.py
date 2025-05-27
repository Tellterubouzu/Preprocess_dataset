import os
import re
import pickle
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
from llm import OpenAI_API
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import string

def batch_iterator(dataset, batch_size):
    batch = []
    for item in dataset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def worker(text, api_key, model, system_prompt_file, max_tokens, temperature):
    llm = OpenAI_API(
        model=model,
        api_key=api_key,
        system_prompt_file=system_prompt_file,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    try:
        resp, tokens = llm.generate_text(text)
        llm.reset_conversation()
        return text, resp, tokens
    except Exception as e:
        print(f"[APIエラー {text[:30]}...]: {e}")
        return None

def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=str, default="thai_text_processed.txt")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--system_prompt_file", type=str, default="./prompts/system_prompt.txt")
    parser.add_argument("--max_tokens", type=int, default=9192)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--num_lines", type=int, default=10000)
    parser.add_argument("--dataset_name", type=str, default="pythainlp/thaigov-v2-corpus-31032024")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=200, help="同時API呼び出し数")
    args = parser.parse_args()

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    cache_path = "cache.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    ds = load_dataset(args.dataset_name, split="train", streaming=True)

    written = 0
    total_tokens_used = 0
    seen_sentences = set()
    lock = threading.Lock()

    def mostly_alnum(text: str, threshold: float = 0.3) -> bool:
        """True なら “英数字率 ≥ threshold”"""
        if not text:
            return False
        alnum_cnt = sum(ch in string.ascii_letters + string.digits for ch in text)
        return alnum_cnt / len(text) >= threshold
    
    def remove_loading_number_and_dot(text):
        pattern = r"^\d{1,2}\.\s*"
        text = re.sub(pattern, "", text)
        pattern = r"^เรื่อง\s*"
        return re.sub(pattern, "", text)

    def contains_long_numeric_pattern(text:str,min_length:int=6) -> bool:
        pattern = rf"[0-9./-]{{{min_length},}}"
        return re.search(pattern, text)

    pbar = tqdm(total=args.num_lines, desc="書き込み行数", unit="line")
    with open(args.output_file, "w", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=args.workers) as executor:

        for batch in batch_iterator(ds, args.batch_size):
            if written >= args.num_lines or total_tokens_used >= args.limit:
                break

            texts = []
            for example in batch:
                txt = example.get("context", "").strip()
                if txt and txt not in cache and not mostly_alnum(txt):
                    texts.append(txt)

            if not texts:
                continue

            future_to_text = {
                executor.submit(
                    worker,
                    text,
                    openai_api_key,
                    args.model,
                    args.system_prompt_file,
                    args.max_tokens,
                    args.temperature
                ): text for text in texts
            }

            for future in as_completed(future_to_text):
                if written >= args.num_lines or total_tokens_used >= args.limit:
                    break

                result = future.result()
                if not result:
                    continue
                orig_text, resp, tokens = result
                cache[orig_text] = [s.strip() for s in resp.split("#")]
                total_tokens_used += tokens

                with lock:
                    for s in cache[orig_text]:
                        if mostly_alnum(s) or len(s) <30:
                            continue
                        if written >= args.num_lines:
                            break   
                        if "อาทิ" in s or "อัตรา" in s or "อัตราการ" in s:
                            continue
                        if s in seen_sentences:
                            continue
                        if contains_long_numeric_pattern(s):
                            continue
                        s = remove_loading_number_and_dot(s)
                        out_f.write(s + "\n")
                        seen_sentences.add(s)
                        written += 1
                        pbar.update(1)
                        pbar.set_postfix(cost=f"${total_tokens_used:.2f}")

            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)

        pbar.close()

    print(f"\n✅ 完了: {written} 文を '{args.output_file}' に保存しました。")
    print(f"💰 推定API料金: ${total_tokens_used}")

if __name__ == "__main__":
    main()