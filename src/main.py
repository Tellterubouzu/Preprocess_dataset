# import os
# import re
# import pickle
# from dotenv import load_dotenv
# from datasets import load_dataset
# from tqdm import tqdm
# from llm import OpenAI_API
# from argparse import ArgumentParser


# def batch_iterator(dataset, batch_size):
#     batch = []
#     for item in dataset:
#         batch.append(item)
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
#     if batch:
#         yield batch

# def main():
#     parser = ArgumentParser()
#     parser.add_argument("--output_file", type=str, default="thai_text_processed.txt")
#     parser.add_argument("--model", type=str, default="gpt-4o-mini")
#     parser.add_argument("--system_prompt_file", type=str, default="./prompts/system_prompt.txt")
#     parser.add_argument("--max_tokens", type=int, default=9192)
#     parser.add_argument("--temperature", type=float, default=0.5)
#     parser.add_argument("--num_lines", type=int, default=10000)
#     parser.add_argument("--dataset_name", type=str, default="pythainlp/thaigov-v2-corpus-31032024")
#     parser.add_argument("--batch_size", type=int, default=300)
#     parser.add_argument("--limit", type=int, default=100)
#     args = parser.parse_args()

#     load_dotenv()
#     openai_api_key = os.getenv("OPENAI_API_KEY")


#     cache_path = "cache.pkl"
#     if os.path.exists(cache_path):
#         with open(cache_path, "rb") as f:
#             cache = pickle.load(f)
#     else:
#         cache = {}

#     ds = load_dataset(
#         args.dataset_name,
#         split="train",
#         streaming=True
#     )

#     written = 0
#     total_tokens_used = 0
#     seen_sentences = set()

#     llm = OpenAI_API(
#         model=args.model,
#         api_key=openai_api_key,
#         system_prompt_file=args.system_prompt_file,
#         max_tokens=args.max_tokens,
#         temperature=args.temperature,
#     )
#     with open(args.output_file, "w", encoding="utf-8") as out_f:
#         # 行数カウント用の tqdm: 総行数 = args.num_lines
#         pbar = tqdm(total=args.num_lines, desc="書き込み行数",unit="line")
#         for batch in batch_iterator(ds, args.batch_size):
#             # 目標行数に達したら完全にループ脱出
#             if written >= args.num_lines:
#                 break

#             prompts = []
#             original_texts = []

#             for example in batch:
#                 text = example.get("context", "").strip()
#                 if not text or text in cache:
#                     continue
#                 original_texts.append(text)
#                 prompts.append(text)

#             if not prompts:
#                 continue

#             for text in prompts:
#                 try:
#                     resp, tokens = llm.generate_text(text)
#                     llm.reset_conversation()
#                 except Exception as e:
#                     print(f"[APIエラー]: {e}")
#                     continue
#                 #sentences = [s.strip() for s in resp.splitlines()]
#                 sentences = [s.strip() for s in resp.split("#")]
                

#                 cache[text] = sentences
#                 total_tokens_used += tokens

#                 for s in sentences:
#                     if written >= args.num_lines:
#                         break
#                     if s in seen_sentences:
#                         continue
#                     out_f.write(s + "\n")
#                     seen_sentences.add(s)
#                     written += 1
#                     pbar.update(1)
#                     pbar.set_postfix(cost=f"${total_tokens_used:.2f}")
            
#             if written >= args.num_lines:
#                 break
#             if total_tokens_used >= args.limit:
#                 break
#             #break
#             with open(cache_path, "wb") as f:
#                 pickle.dump(cache, f)
#         pbar.close()

#     print(f"\n✅ 完了: {written} 文を '{args.output_file}' に保存しました。")
#     print(f"💰 推定API料金: ${total_tokens_used}")

# if __name__ == "__main__":
#     main()

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
    """
    各スレッドで走る関数。新規にOpenAI_APIを作って問い合わせ、
    (元テキスト, レスポンス文字列, トークン数) を返す。
    """
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
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=100, help="同時API呼び出し数")
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

    pbar = tqdm(total=args.num_lines, desc="書き込み行数", unit="line")
    with open(args.output_file, "w", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=args.workers) as executor:

        for batch in batch_iterator(ds, args.batch_size):
            if written >= args.num_lines or total_tokens_used >= args.limit:
                break

            texts = []
            for example in batch:
                txt = example.get("context", "").strip()
                if txt and txt not in cache:
                    texts.append(txt)

            if not texts:
                continue

            # 1) スレッドプールにジョブ投入
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

            # 2) 完了したものから順次処理
            for future in as_completed(future_to_text):
                if written >= args.num_lines or total_tokens_used >= args.limit:
                    break

                result = future.result()
                if not result:
                    continue
                orig_text, resp, tokens = result
                cache[orig_text] = [s.strip() for s in resp.split("#")]
                total_tokens_used += tokens

                # 3) 書き込みはロック下で
                with lock:
                    for s in cache[orig_text]:
                        if written >= args.num_lines:
                            break
                        if s in seen_sentences:
                            continue
                        out_f.write(s + "\n")
                        seen_sentences.add(s)
                        written += 1
                        pbar.update(1)
                        pbar.set_postfix(cost=f"${total_tokens_used:.2f}")

            # キャッシュ保存
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)

        pbar.close()

    print(f"\n✅ 完了: {written} 文を '{args.output_file}' に保存しました。")
    print(f"💰 推定API料金: ${total_tokens_used}")

if __name__ == "__main__":
    main()