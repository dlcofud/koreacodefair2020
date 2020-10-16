# 아래 tf_image 함수는 사용자가 입력한 이미지를 view.py 에서 처리할 수 있도록 하는 함수입니다.


def tf_image(url):
    # 추후 필요여부 담아서 보내줄 리스트 (0: 무필요, 1: 필요 | blue, red, green, orange, pink)
    out = [0, 0, 0, 0, 0]

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_set = test_datagen.flow_from_directory(url[:-12],
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary',
                                                color_mode='grayscale')

    img = Image.open(url)
    data = np.ndarray(shape=(1, 64, 64, 1), dtype=np.float32)
    size = (64, 64)
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    img_array = np.asarray(img)
    img_array = tensorflow.image.rgb_to_grayscale(img_array, name=None)
    #normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = img_array  # normalized_image_array

    # 블루 모델 단계

    blue_model = tensorflow.keras.models.load_model(
        os.path.dirname(__file__) + '/tf/model_blue.h5')
    out_blue = blue_model.predict(data)
    if out_blue > 0.5:
        out[0] = 1

    # 레드 모델 단계

    red_model = tensorflow.keras.models.load_model(
        os.path.dirname(__file__) + '/tf/model_red.h5')
    out_red = red_model.predict(data)
    if out_red > 0.5:
        out[1] = 1

    # 그린 모델 단계

    green_model = tensorflow.keras.models.load_model(
        os.path.dirname(__file__) + '/tf/model_green_2.h5')
    out_green = green_model.predict(data)
    if out_green > 0.5:
        out[2] = 1

    # 오렌지 모델 단계

    orange_model = tensorflow.keras.models.load_model(
        os.path.dirname(__file__) + '/tf/model_orange.h5')
    out_orange = orange_model.predict(data)
    if out_orange > 0.5:
        out[3] = 1

    # 핑크 모델 단계

    pink_model = tensorflow.keras.models.load_model(
        os.path.dirname(__file__) + '/tf/model_pink_edited.h5')
    out_pink = pink_model.predict(data)
    out_pink = np.argmax(out_pink[0])
    out[4] = out_pink

# 아래는 사용자가 선택한 사진을 업로드하고 tf_image를 통해 데이터를 뽑아낸 뒤
# 해당 정보를 가공하여 출력하는 코드입니다.


def upload(request):
    tower_type = request.POST['tower_type']
    floor = request.POST['floor']
    img = request.FILES['plan']

    photo = fileupload()
    photo.image = img
    photo.save()

    re = tf_image("/workspace/study/mydjango/mysite" + photo.image.url)
    # return HttpResponseRedirect(reverse('firepy:index'))

    l1 = []  # 설치해야 할 장치의 종류
    l2 = []  # 관련 법률

    for i in range(0, 5):  # re 값에 따라 필요한 장치를 글로 저장
        if re[i] >= 1:
            if i == 0:
                l1.append("자동소화 설비를 설치해야 합니다.")
                l2.append("아파트 등 30층 이상 오피스텔에 설치해야 합니다.")
            if i == 1:
                l1.append("소화기를 설치해야 합니다.")
                l2.append("연면적 33m² 이상의 모든 건물에 설치해야 합니다.")
            if i == 2:
                l1.append("피난/구조 설비를 설치해야 합니다.")
                l2.append("5층 이상 주택에 유도등, 비상 조명을 설치해야 합니다.")
            if i == 3:
                l1.append("화재 경보 장치를 설치해야 합니다.")
                l2.append("연면적 400m² 이상의 건물에 경보 장치를 설치해야 합니다.")
            if i == 4:
                if re[i] < 4:
                    l1.append("스프링클러를 총 "+str(re[i])+"개 설치해야 합니다.")
                else:
                    l1.append("스프링클러를 4개 이상 설치해야 합니다.")
                #    스프링클러는 예외적으로 '필요 개수' 출력
                l2.append("6층 이상 주거시설에 스프링클러를 설치해야 합니다.")

    context = {'image_url': 'static/firepy/SampleImg.jpg',
               'textList': l1, 'lawList': l2}

    return render(request, 'firepy/result.html', {'image_url': photo.image.url, 'textList': l1, 'lawList': l2})


# 아래 코드는 사용자가 불순한 의도를 가지고 js 파일등을 업로드할 때를 방지하기 위해
# 파일의 이름을 임의로 수정하는 로직입니다.

def pathmaker(instance, filename):
    from random import choice
    import string
    arr = [choice(string.ascii_letters) for _ in range(8)]
    pid = ''.join(arr)
    extension = os.path.splitext(filename)[-1].lower()
    return pid + "/" + pid + extension
