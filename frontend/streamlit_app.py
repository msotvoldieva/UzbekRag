import streamlit as st
import requests

st.title("📚 Mehnat Kodeksi RAG va Arizalar yuklash")

# Single input field for all queries
user_input = st.text_input("Savolingizni bering yoki kerakli ariza nomini kiriting:")

if st.button("Yuborish"):
    if not user_input.strip():
        st.error("Iltimos savolingizni yozing.")
    else:
        try:
            # Send request to intelligent endpoint
            with st.spinner("Tahlil qilinmoqda..."):
                response = requests.post("http://backend:8000/intelligent_query",
                                    json={"query": user_input})
            
            if response.status_code == 200:
                # Check content type to determine response format
                content_type = response.headers.get('content-type', '')
                
                if 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                    # Handle DOCX file download
                    filename = response.headers.get('content-disposition', '').split('filename=')[-1].strip('"')
                    if not filename:
                        filename = f"ariza.docx"
                    
                    st.success(f"📄 Ariza topildi: {filename}")
                    st.download_button(
                        label="📥 Yuklab olish",
                        data=response.content,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    
                elif 'application/json' in content_type:
                    data = response.json()
                    
                    if 'answer' in data:
                        st.markdown("### 💬 Javob:")
                        st.write(data.get('answer'))
                         
                    elif 'documents' in data:
                        # Multiple documents found
                        st.markdown("### 📑 Topilgan hujatlar:")
                        for i, doc in enumerate(data['documents'], 1):
                            st.write(f"{i}. {doc}")
                            
                    elif 'error' in data:
                        st.error(data.get('error'))

                else:
                    st.error("Noma'lum javob turi")
                    
            else:
                try:
                    error_data = response.json()
                    if response.status_code == 404:
                        st.error('🔍 Kerakli fayl yoki ma\'lumot topilmadi!')
                    else:
                        st.error(f"❌ Xato: {response.status_code} - {error_data.get('detail')}, 'Noma\\'lum xato')")
                except:
                    st.error(f"❌ Server xatosi: {response.status_code}")
                    
        except requests.RequestException as e:
            st.error(f"🔗 Ulanishda xato: {str(e)}")



# haystack and vllm, monitoring